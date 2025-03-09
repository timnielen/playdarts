import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
import math
from scipy.optimize import linear_sum_assignment


COST_DARTS = 5
COST_CLASS = 1

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
    
class PositionalEncoding(nn.Module):
    """ Adds positional encoding to the input sequence """
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0,1)  # Shape: (max_len, 1, dim)
        
        
    def forward(self, x):
        return x + self.pe[:x.shape[0], :].to(x.device)
    
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        B, C, W, H = x.shape
        not_mask = torch.ones((B,W,H), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # print(x_embed.shape, y_embed.shape, dim_t.shape) 
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class CNNTransformer(nn.Module):
    def __init__(self, d=256, num_dart_queries=3, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()
        name="resnet34"
        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, False],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        
        train_backbone = True
        for name, parameter in resnet.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        return_interm_layers = False        
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        # Freeze backbone parameters
        # for param in self.backbone.parameters():
        #     param.requires_grad = False  # Freezes the CNN backbone

        # 1x1 Conv to reduce channels from num_channels to d
        self.conv1x1 = nn.Conv2d(num_channels, d, kernel_size=1)

        # Transformer
        self.transformer = nn.Transformer(d_model=d, nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)
        
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        # Positional Encoding
        N_steps = d // 2
        self.positional_encoding = PositionEmbeddingSine(N_steps, normalize=True)

        # Learnable queries (num_queries x d)
        self.queries = nn.Embedding(num_dart_queries+4, d)

        # Final linear layer (shared across all queries)
        self.class_embed = nn.Linear(d, 2)
        num_layers = 3
        output_dim = 2
        self.pos_embed = MLP(d, dim_feedforward, output_dim, num_layers)

    def forward(self, x):
        B, C, H, W = x.shape  # Input: (B, 3, H, W)
        
        # CNN Backbone
        x = self.backbone(x)["0"] # Shape: (B, num_channels, H/32, W/32)
        
        x = self.conv1x1(x)   # Shape: (B, d, H/32, W/32)
        extracted_features = x.clone()
        
        pos = self.positional_encoding(x)
        x = x + pos
        
        # Add positional encoding
        
        # Flatten spatial dimensions: (B, d, H/32 * W/32) → (B, d, H/32 * W/32)
        x = x.reshape(B, x.shape[1], -1)
        # permute: (B, d, H/32 * W/32) → (H/32 * W/32, B, d)
        x = x.permute(2,0,1)

        #broadcast queries to batch
        queries = self.queries.weight.unsqueeze(1).repeat(1, B, 1)
        # Transformer 
        x = self.transformer.encoder(x) # Shape: (7, B, d)
        x = self.transformer.decoder(queries, x)
        
        pred_class = self.class_embed(x[4:]) # Shape (num_dart_queries, B, 2)
        # assert pred_class.shape == (3, B, 2), pred_class.shape
        pred_point = self.pos_embed(x) # Shape (num_dart_queries+4, B, 2)
        # assert pred_point.shape == (7, B, 2), pred_point.shape
        logits = pred_class.transpose(0,1)
        is_dart = F.softmax(logits, dim=-1)
        
        pred_point = pred_point.transpose(0,1).sigmoid()
        corners = pred_point[:, :4]
        darts = pred_point[:, 4:]
        outputs = {"corners": corners, "darts": darts, "is_dart": is_dart, "is_dart_logits": logits, "extracted_features": extracted_features}
        return outputs
    
    

    

    
    
def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def loss_corners(outputs, targets):
    assert 'corners' in outputs
    src_corners = outputs["corners"].reshape(-1, 2)
    tgt_corners = torch.cat([t["corners"] for t in targets])
    assert src_corners.shape == tgt_corners.shape
    loss_corners = F.l1_loss(src_corners, tgt_corners)
    return loss_corners

def loss_darts(outputs, targets, indices):
        assert 'darts' in outputs
        idx = _get_src_permutation_idx(indices)
        pred_darts = outputs['darts'][idx]
        target_darts = torch.cat([t['darts'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_darts = F.l1_loss(pred_darts, target_darts)
        return loss_darts
    
def loss_labels(outputs, indices):
        assert 'is_dart_logits' in outputs
        src_logits = outputs['is_dart_logits']
        
        #create labels for darts, i.e. those outputs that have been matched to a dart get class 1 else 0
        idx = _get_src_permutation_idx(indices)
        target_classes = torch.zeros(src_logits.shape[:2],
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 1
        
        class_weights = torch.ones(2, device=src_logits.device)
        # class_weights[0] = 3/1
        # class_weights[1] = 3/2 
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, class_weights)

        return loss_ce
    
@torch.no_grad()    
def match(outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "is_dart": Tensor of dim [batch_size, num_queries, 2] with the classification probabilities
                 "darts": Tensor of dim [batch_size, num_queries, 2] with the predicted dart coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "darts": Tensor of dim [num_target_boxes, 2] containing the target dart coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["is_dart"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["is_dart"].flatten(0, 1)  # [batch_size * num_queries, 2]
        out_darts = outputs["darts"].flatten(0, 1)  # [batch_size * num_queries, 2]
        
        # Also concat the target labels and boxes
        tgt_darts = torch.cat([v["darts"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, torch.ones(tgt_darts.shape[0], dtype=torch.int32)]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_darts, tgt_darts, p=1)
        # Final cost matrix
        C = COST_DARTS * cost_bbox + COST_CLASS * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["darts"]) for v in targets]
        try:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        except:
            print(C)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    
# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
class DomainDiscriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=1.0)
        
        # Convolutional layers to process the feature maps
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, stride=2, padding=1)  # 256 -> 512 feature maps
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  
        
        # Batch normalization to stabilize training
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(1024)
        
        # Fully connected layers after flattening
        self.fc1 = nn.Linear(1024, hidden_dim)  # Flattened size after conv layers
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output: single logit for binary classification
        
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.grl(x)
        
        # Pass through convolutional layers with batch normalization and ReLU activation
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Flatten the output from convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, 2048)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer
        
        return x
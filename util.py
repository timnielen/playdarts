import torch

def prepare_6_channel(model):
    conv1 = model.model.backbone.conv_encoder.model.conv1  # Conv2d(3, 64, 7, 2, 3)

    # Repeat the weights across channels: from [64, 3, 7, 7] -> [64, 6, 7, 7]
    new_weight = conv1.weight.repeat(1, 2, 1, 1)[:, :6, :, :]  # safely cut to exactly 6

    # Create a new Conv2d layer with 6 input channels
    new_conv1 = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.requires_grad_(False)
        new_conv1.weight.copy_(new_weight)
    model.model.backbone.conv_encoder.model.conv1 = new_conv1  # Replace the original conv1 with the new one
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


import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.image_transforms import (
    PaddingMode,
    pad,
)
from transformers.image_utils import ChannelDimension, get_image_size
def safe_squeeze(arr: np.ndarray, axis = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr

class CustomDetrImageProcessor(DetrImageProcessor):
    def _update_annotation_for_padded_image(
        self,
        annotation,
        input_image_size,
        output_image_size,
        padding,
        update_bboxes,
    ):
        """
        Update the annotation for a padded image.
        """
        new_annotation = {}
        new_annotation["size"] = output_image_size

        for key, value in annotation.items():
            if key == "masks":
                masks = value
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                masks = safe_squeeze(masks, 1)
                new_annotation["masks"] = masks
            elif key == "boxes" and update_bboxes:
                boxes = value
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                boxes += np.asarray(
                    [
                        padding[1][0] / output_image_size[1],
                        padding[0][0] / output_image_size[0],
                        0,
                        0,
                    ]
                )
                new_annotation["boxes"] = boxes
            elif key == "size":
                new_annotation["size"] = output_image_size
            else:
                new_annotation[key] = value
        return new_annotation

    def _pad_image(
        self,
        image: np.ndarray,
        output_size,
        annotation = None,
        constant_values= 0,
        data_format = None,
        input_data_format = None,
        update_bboxes = True,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom, pad_top = int((output_height - input_height)//2), int((output_height - input_height + 1)/2)
        pad_right, pad_left = int((output_width - input_width)/2), int((output_width - input_width + 1)/2)
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        return padded_image, annotation
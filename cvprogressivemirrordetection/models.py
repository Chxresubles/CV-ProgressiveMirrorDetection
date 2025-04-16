import os
import sys
from typing import Optional
import torch
from torch import Tensor, nn
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
    FCN_ResNet50_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "sub", "SAM2_UNet"))
from cvprogressivemirrordetection.sub.SAM2_UNet.SAM2UNet import SAM2UNet as SAM2UNetBase


class FCN(nn.Module):
    def __init__(self, pretrained: Optional[bool] = True) -> None:
        super().__init__()
        # Load FCN with ResNet50 backbone
        self.fcn = fcn_resnet50(
            weights=FCN_ResNet50_Weights.DEFAULT if pretrained else None
        )

        # Modify the classifier to output single channel
        self.fcn.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Get model output
        out = self.fcn(x)["out"]
        if self.training:
            return out
        return torch.sigmoid(out)


class DeepLabV3(nn.Module):
    def __init__(self, pretrained: Optional[bool] = True) -> None:
        super().__init__()
        # Load DeepLabV3 with MobileNet backbone
        self.deeplab = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )

        # Modify the classifier to output single channel
        self.deeplab.classifier[4] = nn.Conv2d(
            256, 1, kernel_size=(1, 1), stride=(1, 1)
        )
        self.deeplab.aux_classifier[4] = nn.Conv2d(
            10, 1, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Get model output
        out = self.deeplab(x)["out"]
        if self.training:
            return out
        return torch.sigmoid(out)


class LRASPP(nn.Module):
    def __init__(self, pretrained: Optional[bool] = True) -> None:
        super().__init__()
        # Load LRASPP with MobileNet backbone
        self.lraspp = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )

        # Modify the classifier to output single channel
        self.lraspp.classifier.low_classifier = nn.Conv2d(
            40, 1, kernel_size=(1, 1), stride=(1, 1)
        )
        self.lraspp.classifier.high_classifier = nn.Conv2d(
            128, 1, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Get model output
        out = self.lraspp(x)["out"]
        if self.training:
            return out
        return torch.sigmoid(out)


class SAM2UNet(nn.Module):
    def __init__(self, weights_path: Optional[str] = None) -> None:
        super().__init__()
        # Load SAM2-UNet
        self.sam2unet = SAM2UNetBase(weights_path)

    def forward(self, x: Tensor) -> Tensor:
        # Get model output
        out = self.sam2unet(x)
        if self.training:
            return out
        return torch.sigmoid(out[0])

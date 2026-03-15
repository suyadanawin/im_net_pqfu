import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_tiny_imagenet(num_classes: int = 200) -> nn.Module:
    """
    ResNet-18 adapted for Tiny-ImageNet-200.
    Tiny-ImageNet images are 64x64, so we use a smaller first conv
    and remove the aggressive maxpool used for ImageNet-1k 224x224 input.
    """
    model = resnet18(weights=None)

    # Better for 64x64 inputs
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    # Final classifier for 200 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
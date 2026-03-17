import torchvision.models as models
import torch.nn as nn


def get_model(model_name: str = "resnet18", num_classes: int = 200, pretrained: bool = False):
    if model_name.lower() != "resnet18":
        raise ValueError(f"Unsupported model: {model_name}. Only resnet18 is supported.")

    if pretrained:
        # Tiny-ImageNet is not ImageNet-1K, but pretrained weights can still be used if desired.
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
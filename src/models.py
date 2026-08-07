"""
CNN model definitions: ResNet-50 and EfficientNet-B0, fine-tuned for
CGR image classification.

Both models are initialized with ImageNet pretrained weights. The final
classification head is replaced for the target number of classes.

Fine-tuning strategy:
  - Phase 1 (frozen backbone): train only the classifier head
  - Phase 2 (unfrozen): fine-tune entire network with low LR

"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Model configuration

@dataclass
class ModelConfig:
    """Configuration for model building and training."""
    model_name: str = "resnet50"        # "resnet50" or "efficientnet_b0"
    n_classes: int = 6                   # 2 for binary, 6 for multiclass
    pretrained: bool = True              # use ImageNet weights
    dropout: float = 0.3                 # dropout before final FC
    freeze_backbone: bool = False        # freeze backbone initially
    label_smoothing: float = 0.1         # for loss function
    # Training hyperparameters
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    lr_patience: int = 5                 # ReduceLROnPlateau patience
    lr_factor: float = 0.5              # LR reduction factor
    early_stop_patience: int = 10


# ResNet-50

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 fine-tuned for CGR image classification.

    Architecture:
        ResNet-50 backbone (pretrained on ImageNet)
        - Global Average Pooling
        - Dropout(p)
        - Linear(2048, n_classes)

    The final residual block (layer4) is always unfrozen for fine-tuning,
    even in frozen backbone mode.
    """

    def __init__(self, n_classes: int, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # 2048

        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, n_classes)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all layers except layer4 and fc."""
        for name, param in self.backbone.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning phase 2."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """Return last conv layer for Grad-CAM."""
        return self.backbone.layer4[-1].conv3

    def get_gradcam_target_layer(self) -> list:
        """Return target layers list for pytorch-grad-cam."""
        return [self.backbone.layer4[-1]]


# EfficientNet-B0

class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for CGR image classification.

    Architecture:
        EfficientNet-B0 backbone (pretrained on ImageNet)
        - Global Average Pooling (built-in)
        - Dropout(p)
        - Linear(1280, n_classes)

    EfficientNet-B0 has 5.3M parameters vs ResNet-50's 25.6M —
    it trains faster and generalizes better on smaller datasets.
    """

    def __init__(self, n_classes: int, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, n_classes)
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze features, keep classifier trainable."""
        for name, param in self.backbone.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> list:
        """Return target layers list for pytorch-grad-cam."""
        # Last MBConv block in EfficientNet-B0 = features[-2]
        return [self.backbone.features[-1]]


# Factory function

def build_model(
    model_name: str,
    n_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a CNN classifier for CGR images.

    Args:
        model_name:       "resnet50" or "efficientnet_b0"
        n_classes:        number of output classes (2 or 6)
        pretrained:       load ImageNet pretrained weights
        dropout:          dropout probability before final FC
        freeze_backbone:  freeze backbone layers initially

    Returns:
        PyTorch model with get_gradcam_target_layer() method
    """
    name = model_name.lower().replace("-", "_")

    if name == "resnet50":
        model = ResNet50Classifier(
            n_classes=n_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    elif name in ("efficientnet_b0", "efficientnet-b0"):
        model = EfficientNetB0Classifier(
            n_classes=n_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: resnet50, efficientnet_b0"
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Model:      {model_name}")
    print(f"  Classes:    {n_classes}")
    print(f"  Parameters: {n_params:.1f}M total, {n_trainable:.1f}M trainable")

    return model


# Loss function

def build_loss_fn(
    task: str = "multiclass",
    n_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    """
    CrossEntropyLoss for both binary and multiclass —
    works with n_classes output neurons in both cases.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )


# Model checkpoint utilities

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    filepath: str,
    config: Optional[ModelConfig] = None,
):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "model_class": type(model).__name__,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, checkpoint.get("metrics", {})

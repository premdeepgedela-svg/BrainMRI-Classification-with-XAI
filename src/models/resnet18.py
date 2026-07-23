import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BrainMRIResNet18(nn.Module):
    """
    ResNet-18 adapted for 4-channel Brain MRI classification.
    """

    def __init__(self, num_classes=2):

        super().__init__()

        # ----------------------------------------
        # Load pretrained ResNet18
        # ----------------------------------------

        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # ----------------------------------------
        # Replace first convolution
        # ----------------------------------------

        old_conv = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # ----------------------------------------
        # Initialize new convolution weights
        # ----------------------------------------

        with torch.no_grad():

            # Copy pretrained RGB weights
            self.model.conv1.weight[:, :3] = old_conv.weight

            # Fourth channel = average of RGB filters
            self.model.conv1.weight[:, 3] = (
                old_conv.weight.mean(dim=1)
            )

        # ----------------------------------------
        # Replace classifier
        # ----------------------------------------

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes,
        )

        # ----------------------------------------
        # Freeze pretrained backbone
        # ----------------------------------------

        for param in self.model.parameters():
            param.requires_grad = False

        # Train first convolution
        for param in self.model.conv1.parameters():
            param.requires_grad = True

        # Train classifier
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):

        return self.model(x)

    def unfreeze_backbone(self):
        """
        Unfreeze entire network for fine-tuning.
        """

        for param in self.model.parameters():
            param.requires_grad = True
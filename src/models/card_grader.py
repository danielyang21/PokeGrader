"""
Neural network models for card grading.
"""

import torch
import torch.nn as nn
from torchvision import models


class CardGraderCNN(nn.Module):
    """
    CNN model for predicting card grades.
    Uses transfer learning with a pre-trained ResNet backbone.
    """

    def __init__(self, num_classes=10, pretrained=True):
        """
        Args:
            num_classes: Number of grade classes (1-10)
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(CardGraderCNN, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Freeze early layers (optional - can unfreeze for fine-tuning)
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class CardGraderMultiTask(nn.Module):
    """
    Multi-task model that predicts:
    1. Overall grade (1-10)
    2. Centering score
    3. Corners score
    4. Edges score
    5. Surface score
    """

    def __init__(self, pretrained=True):
        super(CardGraderMultiTask, self).__init__()

        # Shared backbone
        self.backbone = models.resnet34(pretrained=pretrained)
        num_features = self.backbone.fc.in_features

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Task-specific heads
        self.overall_grade = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # Grades 1-10
        )

        self.centering = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.corners = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.edges = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.surface = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Get predictions from each head
        overall = self.overall_grade(features)
        centering = self.centering(features)
        corners = self.corners(features)
        edges = self.edges(features)
        surface = self.surface(features)

        return {
            'overall_grade': overall,
            'centering': centering,
            'corners': corners,
            'edges': edges,
            'surface': surface
        }


def load_model(model_path, model_class=CardGraderCNN, num_classes=10):
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to saved model weights
        model_class: Model class to instantiate
        num_classes: Number of output classes

    Returns:
        Loaded model
    """
    model = model_class(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

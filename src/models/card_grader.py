"""
Pokemon Card Grading Model using PyTorch.

This model predicts:
1. Corner quality scores (fray, fill, angle) for all 8 corners
2. Edge quality scores (fray, fill) for all 8 edges
3. Overall card grade (GEM MINT, MINT, NEAR MINT, etc.)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Optional


class CornerQualityHead(nn.Module):
    """Predicts quality scores for a single corner from corner image."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()

        self.quality_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # fray, fill, angle
        )

    def forward(self, corner_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            corner_features: [batch, feature_dim] corner image features

        Returns:
            [batch, 3] tensor with [fray, fill, angle] scores (0-1000)
        """
        scores = self.quality_predictor(corner_features)
        # Apply sigmoid and scale to [0, 1000] range
        scores = torch.sigmoid(scores) * 1000
        return scores


class EdgeQualityHead(nn.Module):
    """Predicts quality scores for a single edge from edge image."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()

        self.quality_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # fray, fill
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [batch, feature_dim] edge image features

        Returns:
            [batch, 2] tensor with [fray, fill] scores (0-1000)
        """
        scores = self.quality_predictor(edge_features)
        # Apply sigmoid and scale to [0, 1000] range
        scores = torch.sigmoid(scores) * 1000
        return scores


class CardGradeClassifier(nn.Module):
    """Classifies overall card grade from full card images."""

    def __init__(self, feature_dim: int = 512, num_classes: int = 5):
        super().__init__()

        # Grades: GEM MINT, MINT, NEAR MINT, etc.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # *2 for front + back
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, front_features: torch.Tensor, back_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            front_features: [batch, feature_dim] front card features
            back_features: [batch, feature_dim] back card features

        Returns:
            [batch, num_classes] logits for grade classification
        """
        combined = torch.cat([front_features, back_features], dim=1)
        return self.classifier(combined)


class PokeGraderModel(nn.Module):
    """
    Complete Pokemon card grading model.

    Architecture:
    1. Shared CNN backbone (ResNet50) for feature extraction
    2. Separate heads for corners, edges, and overall grade
    3. Multi-task learning with weighted loss
    """

    def __init__(
        self,
        num_grade_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Shared CNN backbone for feature extraction
        resnet = models.resnet50(pretrained=pretrained)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension from ResNet50
        self.feature_dim = 2048

        # Optionally freeze backbone during initial training
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Dimensionality reduction
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Prediction heads
        self.corner_head = CornerQualityHead(feature_dim=512)
        self.edge_head = EdgeQualityHead(feature_dim=512)
        self.grade_classifier = CardGradeClassifier(feature_dim=512, num_classes=num_grade_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using shared backbone.

        Args:
            images: [batch, 3, H, W] input images

        Returns:
            [batch, 512] reduced feature vectors
        """
        # Extract features with backbone
        features = self.backbone(images)  # [batch, 2048, 1, 1]
        features = features.flatten(1)     # [batch, 2048]

        # Reduce dimensionality
        features = self.feature_reducer(features)  # [batch, 512]

        return features

    def predict_corner(self, corner_image: torch.Tensor) -> torch.Tensor:
        """
        Predict quality scores for a corner image.

        Args:
            corner_image: [batch, 3, H, W] corner image

        Returns:
            [batch, 3] scores for [fray, fill, angle]
        """
        features = self.extract_features(corner_image)
        return self.corner_head(features)

    def predict_edge(self, edge_image: torch.Tensor) -> torch.Tensor:
        """
        Predict quality scores for an edge image.

        Args:
            edge_image: [batch, 3, H, W] edge image

        Returns:
            [batch, 2] scores for [fray, fill]
        """
        features = self.extract_features(edge_image)
        return self.edge_head(features)

    def predict_grade(self, front_image: torch.Tensor, back_image: torch.Tensor) -> torch.Tensor:
        """
        Predict overall card grade from front and back images.

        Args:
            front_image: [batch, 3, H, W] front card image
            back_image: [batch, 3, H, W] back card image

        Returns:
            [batch, num_classes] logits for grade classification
        """
        front_features = self.extract_features(front_image)
        back_features = self.extract_features(back_image)
        return self.grade_classifier(front_features, back_features)

    def forward(
        self,
        corner_images: Optional[torch.Tensor] = None,
        edge_images: Optional[torch.Tensor] = None,
        front_image: Optional[torch.Tensor] = None,
        back_image: Optional[torch.Tensor] = None,
        task: str = 'all'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for different tasks.

        Args:
            corner_images: [batch, 3, H, W] corner images
            edge_images: [batch, 3, H, W] edge images
            front_image: [batch, 3, H, W] front card image
            back_image: [batch, 3, H, W] back card image
            task: 'corners', 'edges', 'grade', or 'all'

        Returns:
            Dictionary with predictions for requested tasks
        """
        outputs = {}

        if task in ['corners', 'all'] and corner_images is not None:
            outputs['corner_scores'] = self.predict_corner(corner_images)

        if task in ['edges', 'all'] and edge_images is not None:
            outputs['edge_scores'] = self.predict_edge(edge_images)

        if task in ['grade', 'all'] and front_image is not None and back_image is not None:
            outputs['grade_logits'] = self.predict_grade(front_image, back_image)

        return outputs


def create_model(
    num_grade_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> PokeGraderModel:
    """
    Factory function to create a PokeGrader model.

    Args:
        num_grade_classes: Number of grade categories (default: 5)
                          [GEM MINT, MINT, NEAR MINT, EXCELLENT, etc.]
        pretrained: Use pretrained ResNet50 weights
        freeze_backbone: Freeze backbone weights during training

    Returns:
        PokeGraderModel instance
    """
    return PokeGraderModel(
        num_grade_classes=num_grade_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def load_model(model_path: str, num_grade_classes: int = 5) -> PokeGraderModel:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to saved model weights
        num_grade_classes: Number of output classes

    Returns:
        Loaded model
    """
    model = create_model(num_grade_classes=num_grade_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

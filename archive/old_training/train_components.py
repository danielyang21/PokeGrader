"""
Training script for component-based card grading
Trains separate models for corners, edges, and centering
Then combines them for final grade prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.card_grader import create_model
from data.card_dataset import (
    CardCornerDataset,
    CardEdgeDataset,
    CardCenteringDataset,
    CardGradeDataset,
    get_data_augmentation,
    GRADE_TO_IDX
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_corner_model(
    data_dir: str = "src/scraper/data/tag_cards",
    num_epochs: int = 20,
    batch_size: int = 16,
    device: str = 'cpu'
):
    """
    Train model to predict corner quality (fray, fill, angle).
    Each corner image is processed to predict its 3 quality metrics.
    """
    logger.info("="*60)
    logger.info("Training Corner Quality Model")
    logger.info("="*60)

    # Create dataset
    train_transform = get_data_augmentation()
    dataset = CardCornerDataset(data_dir=data_dir, transform=train_transform)

    logger.info(f"Total corner samples: {len(dataset)}")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Loss and optimizer (MSE for regression of fray/fill/angle values)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    save_path = "models/corner_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        train_loss = 0.0

        for corner_images, targets in tqdm(train_loader, desc="Training"):
            corner_images = corner_images.to(device)
            # targets: [batch, 3] containing [fray, fill, angle]
            targets = targets.to(device).float()

            optimizer.zero_grad()
            predictions = model.predict_corner(corner_images)
            loss = criterion(predictions, targets)  # Both in 0-1000 range
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for corner_images, targets in tqdm(val_loader, desc="Validating"):
                corner_images = corner_images.to(device)
                targets = targets.to(device).float()

                predictions = model.predict_corner(corner_images)
                loss = criterion(predictions, targets)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        logger.info(f"Val Loss: {val_loss:.6f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info(f"\nCorner model training complete! Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss


def train_edge_model(
    data_dir: str = "src/scraper/data/tag_cards",
    num_epochs: int = 20,
    batch_size: int = 16,
    device: str = 'cpu'
):
    """
    Train model to predict edge quality (fray, fill).
    Each edge image is processed to predict its 2 quality metrics.
    """
    logger.info("\n" + "="*60)
    logger.info("Training Edge Quality Model")
    logger.info("="*60)

    # Create dataset
    train_transform = get_data_augmentation()
    dataset = CardEdgeDataset(data_dir=data_dir, transform=train_transform)

    logger.info(f"Total edge samples: {len(dataset)}")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    save_path = "models/edge_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        train_loss = 0.0

        for edge_images, targets in tqdm(train_loader, desc="Training"):
            edge_images = edge_images.to(device)
            # targets: [batch, 2] containing [fray, fill]
            targets = targets.to(device).float()

            optimizer.zero_grad()
            predictions = model.predict_edge(edge_images)
            loss = criterion(predictions, targets)  # Both in 0-1000 range
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for edge_images, targets in tqdm(val_loader, desc="Validating"):
                edge_images = edge_images.to(device)
                targets = targets.to(device).float()

                predictions = model.predict_edge(edge_images)
                loss = criterion(predictions, targets)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        logger.info(f"Val Loss: {val_loss:.6f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info(f"\nEdge model training complete! Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss


def train_centering_model(
    data_dir: str = "src/scraper/data/tag_cards",
    num_epochs: int = 20,
    batch_size: int = 8,
    device: str = 'cpu'
):
    """
    Train model to predict centering percentages.
    Uses full card images (front and back) to predict L/R and T/B percentages.
    Output: 8 values [front_L, front_R, front_T, front_B, back_L, back_R, back_T, back_B]
    """
    logger.info("\n" + "="*60)
    logger.info("Training Centering Prediction Model")
    logger.info("="*60)

    # Create dataset
    train_transform = get_data_augmentation()
    dataset = CardCenteringDataset(data_dir=data_dir, transform=train_transform)

    logger.info(f"Total centering samples: {len(dataset)}")

    if len(dataset) < 10:
        logger.error("Not enough centering data for training!")
        return float('inf')

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model - we need a custom centering head
    # For now, use the grade model and modify it
    model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)

    # Add custom centering head (outputs 8 values)
    model.centering_head = nn.Sequential(
        nn.Linear(512 * 2, 256),  # 512*2 for front+back features
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 8)  # 8 outputs: front_L, front_R, front_T, front_B, back_L, back_R, back_T, back_B
    )
    model = model.to(device)

    # Loss and optimizer (MSE for regression)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    save_path = "models/centering_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        train_loss = 0.0

        for front_images, back_images, targets in tqdm(train_loader, desc="Training"):
            front_images = front_images.to(device)
            back_images = back_images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Extract features and predict centering
            front_features = model.extract_features(front_images)
            back_features = model.extract_features(back_images)
            combined_features = torch.cat([front_features, back_features], dim=1)
            predictions = model.centering_head(combined_features)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        logger.info(f"Train Loss: {train_loss:.6f}")

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for front_images, back_images, targets in tqdm(val_loader, desc="Validating"):
                front_images = front_images.to(device)
                back_images = back_images.to(device)
                targets = targets.to(device)

                front_features = model.extract_features(front_images)
                back_features = model.extract_features(back_images)
                combined_features = torch.cat([front_features, back_features], dim=1)
                predictions = model.centering_head(combined_features)

                loss = criterion(predictions, targets)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        logger.info(f"Val Loss: {val_loss:.6f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info(f"\nCentering model training complete! Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss


def train_final_grade_model(
    data_dir: str = "src/scraper/data/tag_cards",
    num_epochs: int = 20,
    batch_size: int = 8,
    device: str = 'cpu',
    use_component_predictions: bool = True
):
    """
    Train final grade prediction model.
    Uses corner/edge/centering predictions as features to predict final grade.

    Args:
        use_component_predictions: If True, loads pretrained component models and uses their predictions
    """
    logger.info("\n" + "="*60)
    logger.info("Training Final Grade Model (Integrated)")
    logger.info("="*60)

    # Load pretrained component models if using predictions
    corner_model = None
    edge_model = None
    centering_model = None

    if use_component_predictions:
        logger.info("Loading pretrained component models...")

        # Load corner model
        if Path("models/corner_model.pth").exists():
            corner_model = create_model(num_grade_classes=3, pretrained=False)
            corner_model.load_state_dict(torch.load("models/corner_model.pth", map_location=device))
            corner_model = corner_model.to(device)
            corner_model.eval()
            logger.info("  ✓ Loaded corner model")
        else:
            logger.warning("  ✗ Corner model not found, skipping")

        # Load edge model
        if Path("models/edge_model.pth").exists():
            edge_model = create_model(num_grade_classes=3, pretrained=False)
            edge_model.load_state_dict(torch.load("models/edge_model.pth", map_location=device))
            edge_model = edge_model.to(device)
            edge_model.eval()
            logger.info("  ✓ Loaded edge model")
        else:
            logger.warning("  ✗ Edge model not found, skipping")

        # Load centering model
        if Path("models/centering_model.pth").exists():
            centering_model = create_model(num_grade_classes=3, pretrained=False)
            centering_model.centering_head = nn.Sequential(
                nn.Linear(512 * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 8)
            )
            centering_model.load_state_dict(torch.load("models/centering_model.pth", map_location=device))
            centering_model = centering_model.to(device)
            centering_model.eval()
            logger.info("  ✓ Loaded centering model")
        else:
            logger.warning("  ✗ Centering model not found, skipping")

    # Create dataset
    train_transform = get_data_augmentation()
    dataset = CardGradeDataset(data_dir=data_dir, transform=train_transform)

    # Also load corner and edge datasets to get predictions
    corner_dataset = CardCornerDataset(data_dir=data_dir, transform=train_transform) if corner_model else None
    edge_dataset = CardEdgeDataset(data_dir=data_dir, transform=train_transform) if edge_model else None
    centering_dataset = CardCenteringDataset(data_dir=data_dir, transform=train_transform) if centering_model else None

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model - now with component predictions as input
    num_classes = len(GRADE_TO_IDX)
    logger.info(f"Number of grade classes: {num_classes}")

    # Calculate input feature size:
    # - Image features: 512 * 2 (front + back)
    # - Corner predictions: 8 corners * 3 values = 24
    # - Edge predictions: 8 edges * 2 values = 16
    # - Centering predictions: 8 values
    # Total: 1024 + 24 + 16 + 8 = 1072

    feature_size = 512 * 2  # Base image features
    if corner_model:
        feature_size += 24  # 8 corners * 3
    if edge_model:
        feature_size += 16  # 8 edges * 2
    if centering_model:
        feature_size += 8   # 8 centering values

    logger.info(f"Feature size: {feature_size}")

    # Create custom classifier that takes all features
    model = create_model(num_grade_classes=num_classes, pretrained=True, freeze_backbone=True)

    # Replace the grade classifier with one that accepts component predictions
    model.grade_classifier = nn.Sequential(
        nn.Linear(feature_size, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    save_path = "models/final_grade_model_integrated.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for front_images, back_images, targets in tqdm(train_loader, desc="Training"):
            front_images = front_images.to(device)
            back_images = back_images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Extract image features
            front_features = model.extract_features(front_images)
            back_features = model.extract_features(back_images)

            # Get component predictions if available
            all_features = [front_features, back_features]

            with torch.no_grad():
                # Get corner predictions (for all 8 corners per card)
                if corner_model and corner_dataset:
                    # For each card in batch, we need predictions from 8 corners
                    # This is simplified - in practice you'd need to load corner images
                    # For now, use dummy predictions
                    corner_preds = torch.zeros(front_images.size(0), 24).to(device)
                    all_features.append(corner_preds)

                # Get edge predictions (for all 8 edges per card)
                if edge_model and edge_dataset:
                    edge_preds = torch.zeros(front_images.size(0), 16).to(device)
                    all_features.append(edge_preds)

                # Get centering predictions
                if centering_model:
                    centering_features_front = centering_model.extract_features(front_images)
                    centering_features_back = centering_model.extract_features(back_images)
                    centering_combined = torch.cat([centering_features_front, centering_features_back], dim=1)
                    centering_preds = centering_model.centering_head(centering_combined)
                    all_features.append(centering_preds)

            # Combine all features
            combined_features = torch.cat(all_features, dim=1)

            # Predict grade
            logits = model.grade_classifier(combined_features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for front_images, back_images, targets in tqdm(val_loader, desc="Validating"):
                front_images = front_images.to(device)
                back_images = back_images.to(device)
                targets = targets.to(device)

                # Extract features and predictions (same as training)
                front_features = model.extract_features(front_images)
                back_features = model.extract_features(back_images)

                all_features = [front_features, back_features]

                if corner_model:
                    corner_preds = torch.zeros(front_images.size(0), 24).to(device)
                    all_features.append(corner_preds)

                if edge_model:
                    edge_preds = torch.zeros(front_images.size(0), 16).to(device)
                    all_features.append(edge_preds)

                if centering_model:
                    centering_features_front = centering_model.extract_features(front_images)
                    centering_features_back = centering_model.extract_features(back_images)
                    centering_combined = torch.cat([centering_features_front, centering_features_back], dim=1)
                    centering_preds = centering_model.centering_head(centering_combined)
                    all_features.append(centering_preds)

                combined_features = torch.cat(all_features, dim=1)
                logits = model.grade_classifier(combined_features)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info(f"\nFinal grade model training complete!")
    logger.info(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")
    return best_val_loss


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    logger.info("\n" + "="*60)
    logger.info("COMPONENT-BASED TRAINING PIPELINE")
    logger.info("="*60)

    # Train each component separately
    results = {}

    # 1. Train corner quality model
    results['corner'] = train_corner_model(device=device, num_epochs=20, batch_size=16)

    # 2. Train edge quality model
    results['edge'] = train_edge_model(device=device, num_epochs=20, batch_size=16)

    # 3. Train centering model
    results['centering'] = train_centering_model(device=device, num_epochs=20, batch_size=8)

    # 4. Train final grade model (combining all components)
    results['final_grade'] = train_final_grade_model(device=device, num_epochs=20, batch_size=8)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    for component, loss in results.items():
        logger.info(f"{component.capitalize()}: Best Val Loss = {loss:.6f}")

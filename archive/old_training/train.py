"""
Training script for Pokemon card grading model - SIMPLE DEMO VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.card_grader import create_model
from data.card_dataset import (
    CardCornerDataset,
    CardEdgeDataset,
    CardGradeDataset,
    get_data_augmentation,
    GRADE_TO_IDX
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_grade_model(
    data_dir: str = "src/scraper/data/tag_cards",
    num_epochs: int = 15,
    batch_size: int = 8,
    device: str = 'cpu'
):
    """Train overall grade classification model."""
    logger.info("="*60)
    logger.info("Training Grade Classification Model")
    logger.info("="*60)

    # Create datasets
    train_transform = get_data_augmentation()
    dataset = CardGradeDataset(data_dir=data_dir, transform=train_transform)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    num_classes = len(set([sample['grade_idx'] for sample in dataset.samples]))
    logger.info(f"Number of grade classes: {num_classes}")

    model = create_model(num_grade_classes=num_classes, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    save_path = "models/grade_model.pth"
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
            logits = model.predict_grade(front_images, back_images)
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

                logits = model.predict_grade(front_images, back_images)
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

    logger.info(f"\nGrade model training complete! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Train grade classification model
    train_grade_model(device=device, num_epochs=15, batch_size=8)

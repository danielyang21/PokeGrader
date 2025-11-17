"""
Training script for card grading model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)

    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate model.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on

    Returns:
        Validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total

    return val_loss, val_acc


def train_model(model, train_loader, val_loader, num_epochs=50,
                learning_rate=0.001, device='cuda', save_dir='checkpoints'):
    """
    Complete training pipeline.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints

    Returns:
        Training history
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')

    print(f"Training on {device}")
    print(f"Total epochs: {num_epochs}")
    print("-" * 60)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                      os.path.join(save_dir, 'best_model.pth'))
            print("Saved best model!")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    print("\nTraining complete!")
    return history


if __name__ == "__main__":
    # Example usage
    from src.models.card_grader import CardGraderCNN
    from src.data.dataset import create_data_loaders

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders (you'll need to prepare your data first)
    # train_loader, val_loader = create_data_loaders(
    #     'data/train.csv', 'data/val.csv', 'data/raw'
    # )

    # Create model
    model = CardGraderCNN(num_classes=10)

    # Train model
    # history = train_model(
    #     model, train_loader, val_loader,
    #     num_epochs=50, learning_rate=0.001, device=device
    # )

    print("Set up complete! Prepare your dataset to start training.")

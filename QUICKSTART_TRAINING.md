# PokeGrader Training Quickstart Guide

## Clean Slate Setup Complete

All old training scripts, models, and logs have been archived to:
- `archive/old_training/` - Old training scripts
- `archive/old_models/` - Old model checkpoints (including broken centering model)
- `archive/old_logs/` - Old training logs

## What We Kept

### Scraping Infrastructure
- `src/scraper/` - All scraping code (cert_number_finder.py, etc.)
- `data/cert_numbers.json` - Your 66 valid cert numbers
- `data/` - Cached card images and metadata

### Model Architecture
- `src/models/card_grader.py` - Model definitions (already correct)

### Useful Tools
- `test_card.py` - Test models on your own images
- `visualize_extractions.py` - Verify corner/edge extraction quality
- `TRAINING_LESSONS_LEARNED.md` - All architecture insights
- `ARCHITECTURE_REDESIGN.md` - Original problem analysis

## Writing Your New Training Script

### Location
Create: `src/training/train.py`

### Key Components Needed

#### 1. Data Loading
```python
import json
from pathlib import Path
from PIL import Image
from src.scraper.tag_scraper import scrape_card_metadata

# Load cert numbers
with open('data/cert_numbers.json', 'r') as f:
    cert_numbers = json.load(f)

# For each cert, scrape metadata
metadata = scrape_card_metadata(cert_number)
# Returns dict with: front_url, back_url, corners, edges, centering, etc.
```

#### 2. Image Extraction Functions
```python
def extract_corner_images(front_img: Image, back_img: Image, corner_size=100):
    """
    Extract 8 corner crops (4 front + 4 back).
    Returns: List of 8 PIL Images (100x100px each)

    Order: TL, TR, BL, BR (front), TL, TR, BL, BR (back)
    """
    corners = []
    for img in [front_img, back_img]:
        w, h = img.size
        corners.append(img.crop((0, 0, corner_size, corner_size)))  # TL
        corners.append(img.crop((w-corner_size, 0, w, corner_size)))  # TR
        corners.append(img.crop((0, h-corner_size, corner_size, h)))  # BL
        corners.append(img.crop((w-corner_size, h-corner_size, w, h)))  # BR
    return corners

def extract_edge_images(front_img: Image, back_img: Image, edge_width=50):
    """
    Extract 8 edge strips (4 front + 4 back).
    Returns: List of 8 PIL Images (edges)

    Order: Top, Right, Bottom, Left (front), Top, Right, Bottom, Left (back)
    """
    edges = []
    for img in [front_img, back_img]:
        w, h = img.size
        edges.append(img.crop((0, 0, w, edge_width)))  # Top
        edges.append(img.crop((w-edge_width, 0, w, h)))  # Right
        edges.append(img.crop((0, h-edge_width, w, h)))  # Bottom
        edges.append(img.crop((0, 0, edge_width, h)))  # Left
    return edges
```

#### 3. Target Parsing
```python
def parse_corner_targets(metadata):
    """
    Parse corner scores from metadata.
    Returns: Tensor of shape [8, 3] for 8 corners × [fray, fill, angle]
    """
    corner_scores = []
    for key in ['front_TL', 'front_TR', 'front_BL', 'front_BR',
                'back_TL', 'back_TR', 'back_BL', 'back_BR']:
        fray = metadata[f'corner_{key}_fray']
        fill = metadata[f'corner_{key}_fill']
        angle = metadata[f'corner_{key}_angle']
        corner_scores.append([fray, fill, angle])
    return torch.tensor(corner_scores, dtype=torch.float32)

def parse_edge_targets(metadata):
    """
    Parse edge scores from metadata.
    Returns: Tensor of shape [8, 2] for 8 edges × [fray, fill]
    """
    edge_scores = []
    for key in ['front_top', 'front_right', 'front_bottom', 'front_left',
                'back_top', 'back_right', 'back_bottom', 'back_left']:
        fray = metadata[f'edge_{key}_fray']
        fill = metadata[f'edge_{key}_fill']
        edge_scores.append([fray, fill])
    return torch.tensor(edge_scores, dtype=torch.float32)

def parse_centering_targets(metadata):
    """
    Parse centering percentages from metadata.
    Returns: Tensor of shape [4] for [front_L%, front_T%, back_L%, back_T%]
             Values normalized to 0-1 range (will be multiplied by 100 for %)

    Example: "47L/53R" → extract 47, normalize to 0.47
    """
    # Front L/R (e.g., "47L/53R")
    lr_str = metadata['centering_front_lr']
    front_l = float(lr_str.split('L')[0]) / 100.0

    # Front T/B (e.g., "48T/52B")
    tb_str = metadata['centering_front_tb']
    front_t = float(tb_str.split('T')[0]) / 100.0

    # Back L/R
    lr_str = metadata['centering_back_lr']
    back_l = float(lr_str.split('L')[0]) / 100.0

    # Back T/B
    tb_str = metadata['centering_back_tb']
    back_t = float(tb_str.split('T')[0]) / 100.0

    return torch.tensor([front_l, front_t, back_l, back_t], dtype=torch.float32)
```

#### 4. Model Creation
```python
from src.models.card_grader import create_model
import torch
import torch.nn as nn

device = 'mps'  # or 'cuda' or 'cpu'

# Corner Model
corner_model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)
corner_model = corner_model.to(device)
corner_optimizer = torch.optim.Adam(corner_model.parameters(), lr=0.0001)

# Edge Model
edge_model = create_model(num_grade_classes=2, pretrained=True, freeze_backbone=True)
edge_model = edge_model.to(device)
edge_optimizer = torch.optim.Adam(edge_model.parameters(), lr=0.0001)

# Centering Model (SPECIAL - needs concatenated features)
centering_model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)
centering_model.centering_head = nn.Sequential(
    nn.Linear(512 * 2, 256),  # 512 features from front + 512 from back
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 4),  # Only 4 outputs!
    nn.Sigmoid()  # Constrain to 0-1 range
).to(device)
centering_model = centering_model.to(device)
centering_optimizer = torch.optim.Adam(centering_model.parameters(), lr=0.0001)

# Loss function
criterion = nn.MSELoss()
```

#### 5. Image Transforms
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
```

#### 6. Training Loop Structure
```python
num_epochs = 10

for epoch in range(num_epochs):
    corner_losses = []
    edge_losses = []
    centering_losses = []

    for cert_number in cert_numbers:
        # 1. Scrape metadata
        metadata = scrape_card_metadata(cert_number)

        # 2. Load images
        front_img = Image.open(metadata['front_image_path']).convert('RGB')
        back_img = Image.open(metadata['back_image_path']).convert('RGB')

        # 3. Extract regions
        corner_crops = extract_corner_images(front_img, back_img)
        edge_crops = extract_edge_images(front_img, back_img)

        # 4. Transform to tensors
        corner_batch = torch.stack([transform(c) for c in corner_crops]).to(device)  # [8, 3, 224, 224]
        edge_batch = torch.stack([transform(e) for e in edge_crops]).to(device)  # [8, 3, 224, 224]
        front_full = transform(front_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        back_full = transform(back_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # 5. Get targets
        corner_targets = parse_corner_targets(metadata).to(device)  # [8, 3]
        edge_targets = parse_edge_targets(metadata).to(device)  # [8, 2]
        centering_targets = parse_centering_targets(metadata).unsqueeze(0).to(device)  # [1, 4]

        # 6. Forward pass - Corners
        corner_optimizer.zero_grad()
        corner_preds = corner_model.predict_corner(corner_batch)  # [8, 3]
        corner_loss = criterion(corner_preds, corner_targets)
        corner_loss.backward()
        corner_optimizer.step()
        corner_losses.append(corner_loss.item())

        # 7. Forward pass - Edges
        edge_optimizer.zero_grad()
        edge_preds = edge_model.predict_edge(edge_batch)  # [8, 2]
        edge_loss = criterion(edge_preds, edge_targets)
        edge_loss.backward()
        edge_optimizer.step()
        edge_losses.append(edge_loss.item())

        # 8. Forward pass - Centering
        centering_optimizer.zero_grad()
        front_features = centering_model.extract_features(front_full)  # [1, 512]
        back_features = centering_model.extract_features(back_full)  # [1, 512]
        combined = torch.cat([front_features, back_features], dim=1)  # [1, 1024]
        centering_preds = centering_model.centering_head(combined)  # [1, 4]
        centering_loss = criterion(centering_preds, centering_targets)
        centering_loss.backward()
        centering_optimizer.step()
        centering_losses.append(centering_loss.item())

    # Print epoch stats
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Corner Loss: {sum(corner_losses)/len(corner_losses):.4f}")
    print(f"  Edge Loss: {sum(edge_losses)/len(edge_losses):.4f}")
    print(f"  Centering Loss: {sum(centering_losses)/len(centering_losses):.4f}")

# Save models
torch.save(corner_model.state_dict(), 'models/corner_model.pth')
torch.save(edge_model.state_dict(), 'models/edge_model.pth')
torch.save(centering_model.state_dict(), 'models/centering_model.pth')
print("Models saved!")
```

## Critical Details

### Centering Model Architecture
The centering model is DIFFERENT from corner/edge models:
1. Uses FULL card images (not crops)
2. Extracts features from both front and back
3. Concatenates features: 512 (front) + 512 (back) = 1024
4. Passes through centering_head (1024 → 4 outputs)
5. Outputs are [front_L%, front_T%, back_L%, back_T%] in range 0-1
6. RIGHT% = 100 - LEFT%, BOTTOM% = 100 - TOP% (computed in test script)

### Model Methods
- `model.predict_corner(tensor)` - For corner predictions
- `model.predict_edge(tensor)` - For edge predictions
- `model.extract_features(tensor)` - Get 512-dim feature vector for centering
- Use `model.centering_head(features)` - For centering predictions

### Testing After Training
```bash
python test_card.py --front ~/Desktop/front.png --back ~/Desktop/back.png
```

This will verify:
- Centering adds to 100% (L+R=100, T+B=100) ✓
- Corner scores are reasonable
- Edge scores are reasonable

## Common Issues to Avoid

1. Don't predict 8 centering values - only predict 4!
2. Don't forget Sigmoid activation on centering head
3. Don't train on full cards for corners/edges - use crops!
4. Don't forget to normalize centering targets to 0-1 range
5. Don't forget ImageNet normalization for image transforms

## Next Steps

1. Write `src/training/train.py` yourself using this guide
2. Train on 66 valid certs
3. Test on your card images
4. Verify centering math works
5. Scale up to more data if needed

Good luck! All the hard architectural decisions are documented in `TRAINING_LESSONS_LEARNED.md`.

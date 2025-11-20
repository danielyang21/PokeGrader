# PokeGrader Architecture Redesign

## Current Problems

1. **Broken Centering Logic**: Predictions don't sum to 100%
2. **Wrong Input Data**: Using full cards to predict corner/edge scores
3. **No Spatial Awareness**: Model doesn't know where corners/edges are
4. **Training on Wrong Level**: Should train on individual corners/edges, not full cards

## Proposed Architecture

### Phase 1: Centering Model (Simplest - Start Here)

**Input**: Full front/back card images
**Output**: 4 values that MUST sum correctly
- Front: left% (right% = 100 - left%)
- Front: top% (bottom% = 100 - top%)
- Back: left% (right% = 100 - left%)
- Back: top% (bottom% = 100 - top%)

**Architecture**:
```
Image → CNN Backbone → 4 outputs (each 0-100)
- Output 1: Front L/R ratio (sigmoid * 100)
- Output 2: Front T/B ratio (sigmoid * 100)
- Output 3: Back L/R ratio (sigmoid * 100)
- Output 4: Back T/B ratio (sigmoid * 100)
```

**Loss**: MSE on the 4 values
**Data**: TAG provides exact percentages (e.g., "47L/53R 33T/67B")

### Phase 2: Corner Quality Model

**Problem with Current Approach**:
- We're using full card images to predict 8 corner scores
- Model has no idea WHERE to look for each corner

**Better Approach - Option A (Region-Based)**:
1. Crop 8 corner regions from front/back images
2. Train model on individual corner crops
3. Each corner image → [fray, fill, angle] prediction

**Better Approach - Option B (Detection-Based)**:
1. Use object detection to locate corners
2. Extract features from each corner region
3. Predict scores for each detected corner

**Architecture for Option A**:
```
For each corner crop (224x224):
  Corner Image → ResNet50 → [fray, fill, angle]

Training:
  - Input: 8 corner crops per card
  - Output: 8 × 3 = 24 scores
  - Loss: MSE on each score
```

### Phase 3: Edge Quality Model

**Same Problem**: Full card → 8 edge scores makes no sense

**Better Approach**:
1. Crop 8 edge regions (vertical/horizontal strips)
2. Train on individual edge crops
3. Each edge image → [fray, fill] prediction

**Architecture**:
```
For each edge crop (edge-shaped rectangle):
  Edge Image → ResNet50 → [fray, fill]

Training:
  - Input: 8 edge crops per card
  - Output: 8 × 2 = 16 scores
  - Loss: MSE on each score
```

## Recommended Implementation Order

### Step 1: Fix Centering (Easiest)
- Use full card images (current approach is OK here)
- Change output layer to predict 4 values with proper constraints
- This should work reasonably well with current data

### Step 2: Create Corner/Edge Cropping Pipeline
- Write code to extract corner regions (e.g., 100x100 crops)
- Write code to extract edge regions (e.g., 400x50 strips)
- Build dataset of cropped regions with their scores

### Step 3: Train Corner Model
- Train on individual corner crops
- Much clearer task: "given this corner, rate its quality"

### Step 4: Train Edge Model
- Train on individual edge crops
- Similar to corners

## Alternative: Multi-Task with Attention

If we want to keep using full images:

```
Full Card Image → CNN Backbone → Attention Module
                                     ↓
                          [attends to 8 corners]
                                     ↓
                          8 corner feature vectors
                                     ↓
                          8 × [fray, fill, angle]
```

This is more complex but could work if we:
1. Add spatial attention mechanisms
2. Force the model to look at corner regions
3. Need large amounts of data

## Key Insights

1. **Spatial Structure Matters**: Cards have specific regions (corners, edges, center)
2. **Hierarchical Prediction**: Should predict region-by-region, not all at once
3. **Constrained Outputs**: Centering percentages must sum to 100
4. **Data Granularity**: TAG gives us individual scores - use them properly!

## What's Wrong with Current Training

```python
# CURRENT (WRONG):
full_card_image → model → [corner_fray, corner_fill, corner_angle]
# Problem: Which corner? Model has no idea!

# CORRECT:
front_TL_corner_crop → model → [TL_fray, TL_fill, TL_angle]
front_TR_corner_crop → model → [TR_fray, TR_fill, TR_angle]
... (8 corners total)
```

## Next Steps

1. Start with centering fix (quick win)
2. Build corner/edge cropping pipeline
3. Retrain with proper spatial structure
4. Consider attention mechanisms later if needed

# Training Lessons Learned

## Architecture Insights

### What We Got Right

1. **Region-Based Training**: Training on cropped corners and edges instead of full cards
   - Extract 100x100px corner crops (4 per side = 8 total)
   - Extract 50px edge strips (4 per side = 8 total)
   - This gives the model spatial awareness of WHERE to look
   - Much better than expecting model to find corners in full card image

2. **Component-Based Approach**: Separate models for different grading aspects
   - Corner Model: Predicts fray, fill, angle for corners
   - Edge Model: Predicts fray, fill for edges
   - Centering Model: Predicts L/R and T/B percentages
   - Each model can specialize in its specific task

3. **Transfer Learning**: Using pretrained ResNet50 backbone
   - Start with ImageNet weights
   - Freeze backbone during initial training
   - Only train task-specific heads
   - Much faster convergence with limited data

### What We Fixed

1. **Centering Mathematical Constraints**
   - **PROBLEM**: Model predicted 8 independent values (front_L, front_R, front_T, front_B, back_L, back_R, back_T, back_B) with no guarantee they sum to 100%
   - **Result**: Got nonsense predictions like 41.4L/44.8R (doesn't add to 100!)
   - **SOLUTION**: Predict only 4 values with Sigmoid activation:
     - front_L% (front_R% = 100 - front_L%)
     - front_T% (front_T% = 100 - front_T%)
     - back_L% (back_R% = 100 - back_L%)
     - back_T% (back_B% = 100 - back_T%)
   - This mathematically guarantees L + R = 100 and T + B = 100

2. **Centering Network Architecture**
   ```python
   # OLD (BROKEN):
   nn.Linear(128, 8)  # 8 unconstrained outputs

   # NEW (FIXED):
   nn.Linear(128, 4),  # Only 4 outputs
   nn.Sigmoid()        # Constrain to 0-1 range (multiply by 100 for percentage)
   ```

3. **Extraction Parameters**
   - 100px corners work well for typical card resolutions (400-600px wide)
   - 50px edges work well (aspect ratio stays reasonable when resized to 224x224)
   - Verified with visualization tool that extractions capture correct regions

## Training Best Practices

### Data Considerations

1. **Quality Over Quantity**:
   - 66 valid cert numbers with complete data > 500 cards with missing scores
   - Ensure all required fields are present (corners, edges, centering)

2. **Data Format**:
   - Centering: "47L/53R" and "48T/52B" format
   - Corners: 8 scores (fray, fill, angle per corner)
   - Edges: 8 scores (fray, fill per edge)

3. **Normalization**:
   - Normalize centering percentages to 0-1 range for training
   - Use ImageNet normalization for images (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Architecture

1. **Backbone**: ResNet50 pretrained on ImageNet
   - Remove final classification layer
   - Extract 512-dimensional feature vector from avgpool layer

2. **Corner Head**:
   ```python
   features (512) → Linear(512, 256) → ReLU → Dropout(0.3)
                  → Linear(256, 128) → ReLU → Dropout(0.2)
                  → Linear(128, 3)    # [fray, fill, angle]
   ```

3. **Edge Head**:
   ```python
   features (512) → Linear(512, 256) → ReLU → Dropout(0.3)
                  → Linear(256, 128) → ReLU → Dropout(0.2)
                  → Linear(128, 2)    # [fray, fill]
   ```

4. **Centering Head**:
   ```python
   concat_features (1024) → Linear(1024, 256) → ReLU → Dropout(0.3)
                          → Linear(256, 128) → ReLU → Dropout(0.2)
                          → Linear(128, 4)    # [front_L%, front_T%, back_L%, back_T%]
                          → Sigmoid()         # Constrain to 0-1
   ```

### Training Parameters

1. **Batch Size**: 10 (for 66 cards, gives ~7 batches per epoch)
2. **Learning Rate**: 0.0001 (Adam optimizer)
3. **Device**: MPS (Mac) or CUDA (GPU) preferred, CPU as fallback
4. **Loss**: MSE (Mean Squared Error) for all regression tasks
5. **Epochs**: Start with 10-20, monitor for convergence

### Training Strategy

1. **Sequential Training**:
   - Train corner model first
   - Train edge model second
   - Train centering model last
   - OR train all three simultaneously with separate losses

2. **Model Checkpointing**:
   - Save models after each training run
   - Keep old models as backups (e.g., model_old.pth)
   - Use descriptive names (corner_model.pth, edge_model.pth, centering_model.pth)

3. **Online Training** (if needed):
   - Scrape card → Extract regions → Train → Update models
   - Useful for continuous learning on new data
   - Can lead to overfitting if not careful

## Common Pitfalls to Avoid

1. **Don't predict complementary values independently**
   - WRONG: Predict both L% and R% separately
   - RIGHT: Predict L%, compute R% = 100 - L%

2. **Don't train on full cards for spatial tasks**
   - WRONG: Full card → predict 8 corner scores
   - RIGHT: 8 corner crops → predict score for each

3. **Don't mix up tensor shapes**
   - Corner batch: [batch_size × 8, 3, 224, 224] (8 corners per card)
   - Edge batch: [batch_size × 8, 3, 224, 224] (8 edges per card)
   - Centering: [batch_size, 3, 224, 224] per image (front and back separately)

4. **Don't forget to normalize inputs**
   - Always use ImageNet normalization for ResNet models
   - Convert centering targets to 0-1 range, not 0-100

5. **Don't skip validation**
   - Test on real card images periodically
   - Verify centering adds to 100%
   - Check corner/edge scores are in valid range (0-1000)

## Recommended Project Structure

```
src/
  models/
    card_grader.py          # Model architecture definitions
  training/
    train.py                # Main training script (CLEAN VERSION)
    stream_trainer.py       # Online training (backup)
  scraper/
    <keep all scraping code>

data/
  cert_numbers.json         # Valid cert numbers
  <cached card images and metadata>

models/
  corner_model.pth          # Trained corner model
  edge_model.pth            # Trained edge model
  centering_model.pth       # Trained centering model (4 outputs!)

tools/
  test_card.py              # Test model on custom images
  visualize_extractions.py  # Verify corner/edge extraction
```

## Next Steps for Clean Slate

1. Create new `src/training/train_clean.py` with lessons learned
2. Backup old models to `models/archive/`
3. Remove old training logs
4. Train from scratch on 66 valid certs with fixed architecture
5. Test on your own card images
6. Scale up to more data if results are good

## Key Metrics to Track

- **Training Loss**: Should decrease steadily
- **Centering Predictions**: L+R should equal 100, T+B should equal 100
- **Score Ranges**: Corner/edge scores should be in [0, 1000] range
- **Convergence**: Loss should stabilize after 10-20 epochs

## Testing Checklist

After training, verify:
- [ ] Centering adds to 100% (L+R=100, T+B=100)
- [ ] Corner scores are reasonable (0-1000 range)
- [ ] Edge scores are reasonable (0-1000 range)
- [ ] Model works on unseen card images
- [ ] Predictions are consistent across multiple runs

# PokeGrader - AI Card Grading System

An AI-powered system to grade Pokemon card conditions and predict PSA/BGS grades using computer vision and deep learning.

## Project Overview

PokeGrader helps collectors:
- Estimate potential PSA/BGS grades before professional submission
- Identify centering issues, edge wear, corner damage, and surface scratches
- Calculate whether grading costs are worthwhile based on potential value
- Detect counterfeit cards

## Tech Stack

- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV, torchvision
- **Backend**: FastAPI
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## Getting Started

### 1. Set Up Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

```
PokeGrader/
├── data/                   # Dataset storage
│   ├── raw/               # Original images
│   ├── processed/         # Preprocessed images
│   └── labels.csv         # Grade labels
├── models/                # Model architectures
├── notebooks/             # Jupyter notebooks for experimentation
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model definitions
│   ├── training/         # Training scripts
│   └── inference/        # Prediction scripts
├── api/                   # FastAPI backend
└── tests/                # Unit tests
```

### 3. Learning Path

**Phase 1: PyTorch Basics** (Week 1-2)
- Tensor operations
- Automatic differentiation
- Building simple neural networks
- Start with `notebooks/01_pytorch_basics.ipynb`

**Phase 2: Computer Vision Fundamentals** (Week 3-4)
- Image preprocessing
- Data augmentation
- CNN architectures (ResNet, EfficientNet)
- Transfer learning
- Work through `notebooks/02_image_classification.ipynb`

**Phase 3: Data Collection** (Week 5-6)
- Scrape graded card images from eBay, PWCC
- Build dataset with grade labels
- Create train/val/test splits
- Use `src/data/scraper.py`

**Phase 4: Model Development** (Week 7-10)
- Start with pre-trained models (transfer learning)
- Fine-tune on card grading dataset
- Multi-task learning: grade prediction + defect detection
- Experiment in `notebooks/03_card_grading_model.ipynb`

**Phase 5: Production** (Week 11-12)
- Build FastAPI backend
- Create simple web interface
- Deploy model

## Data Sources

- **eBay**: Search for "PSA 10 Pokemon card", "PSA 9 Pokemon card", etc.
- **PWCC Marketplace**: High-quality graded card images
- **PSA CardFacts**: Official PSA population data
- **Beckett Grading**: BGS graded examples

## Key Features to Build

1. **Grade Prediction**: Overall condition grade (1-10)
2. **Centering Analysis**: Measure border ratios
3. **Corner Detection**: Identify wear and whitening
4. **Edge Analysis**: Detect edge wear
5. **Surface Inspection**: Find scratches, dents, print lines
6. **Authenticity Check**: Compare against known genuine examples

## Success Metrics

- Accuracy within ±0.5 grades of professional grading
- 90%+ precision on counterfeit detection
- Sub-2 second inference time
- User satisfaction and cost savings

## Next Steps

1. Complete PyTorch basics tutorial
2. Collect initial dataset of 100 cards across different grades
3. Build baseline CNN classifier
4. Iterate and improve

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Computer Vision Course](https://www.coursera.org/learn/convolutional-neural-networks)
- [PSA Grading Standards](https://www.psacard.com/resources/gradingstandards)
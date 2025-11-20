"""
Simple script to test the trained model on your own card images.
Upload front and back images to get grade predictions.
"""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent))
from src.models.card_grader import create_model


class CardGrader:
    """Test the trained model on custom images."""

    def __init__(self, device='mps'):
        self.device = device

        # Load models
        print("Loading trained models...")
        self.corner_model = self._load_model('models/corner_model_online.pth', num_outputs=3, is_centering=False)
        self.edge_model = self._load_model('models/edge_model_online.pth', num_outputs=3, is_centering=False)
        self.centering_model = self._load_model('models/centering_model_online.pth', num_outputs=8, is_centering=True)

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Models loaded successfully!")

    def _load_model(self, path, num_outputs, is_centering=False):
        """Load a trained model."""
        # Centering model needs special handling
        if is_centering:
            # Load with 3 outputs first (to match the saved state dict)
            model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True)

            # Add centering head (4 outputs: front_L%, front_T%, back_L%, back_T%)
            model.centering_head = nn.Sequential(
                nn.Linear(512 * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 4),  # 4 outputs instead of 8
                nn.Sigmoid()  # Constrain to 0-1 range
            ).to(self.device)
        else:
            model = create_model(num_grade_classes=num_outputs, pretrained=True, freeze_backbone=True)

        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def predict(self, front_image_path, back_image_path):
        """
        Predict grades for a card given front and back images.

        Args:
            front_image_path: Path to front image
            back_image_path: Path to back image

        Returns:
            Dictionary with predictions for corners, edges, and centering
        """
        print(f"\nAnalyzing card...")
        print(f"  Front: {front_image_path}")
        print(f"  Back: {back_image_path}")

        # Load and transform images
        front_img = Image.open(front_image_path).convert('RGB')
        back_img = Image.open(back_image_path).convert('RGB')

        front_tensor = self.transform(front_img).unsqueeze(0).to(self.device)
        back_tensor = self.transform(back_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Corner predictions (3 outputs: fray, fill, angle)
            corner_pred = self.corner_model.predict_corner(front_tensor).cpu().numpy()[0]

            # Edge predictions (2 outputs: fray, fill)
            edge_pred = self.edge_model.predict_edge(front_tensor).cpu().numpy()[0]

            # Centering predictions (4 outputs: L%, T% for front and back)
            # Compute R% = 100 - L%, B% = 100 - T%
            front_features = self.centering_model.extract_features(front_tensor)
            back_features = self.centering_model.extract_features(back_tensor)
            centering_pred_raw = self.centering_model.centering_head(
                torch.cat([front_features, back_features], dim=1)
            ).cpu().numpy()[0]

            # centering_pred_raw is [front_L, front_T, back_L, back_T] in range 0-1
            # Convert to percentages and compute R/B values
            front_l = centering_pred_raw[0] * 100
            front_r = 100 - front_l
            front_t = centering_pred_raw[1] * 100
            front_b = 100 - front_t
            back_l = centering_pred_raw[2] * 100
            back_r = 100 - back_l
            back_t = centering_pred_raw[3] * 100
            back_b = 100 - back_t

            centering_pred = [front_l, front_r, front_t, front_b, back_l, back_r, back_t, back_b]

        # Format results
        results = {
            'corners': {
                'fray': float(corner_pred[0]),
                'fill': float(corner_pred[1]),
                'angle': float(corner_pred[2])
            },
            'edges': {
                'fray': float(edge_pred[0]),
                'fill': float(edge_pred[1]),
            },
            'centering': {
                'front_left': float(centering_pred[0]),
                'front_right': float(centering_pred[1]),
                'front_top': float(centering_pred[2]),
                'front_bottom': float(centering_pred[3]),
                'back_left': float(centering_pred[4]),
                'back_right': float(centering_pred[5]),
                'back_top': float(centering_pred[6]),
                'back_bottom': float(centering_pred[7])
            }
        }

        return results

    def print_results(self, results):
        """Pretty print the results."""
        print("\n" + "="*60)
        print("GRADING RESULTS")
        print("="*60)

        print("\nCORNERS:")
        print(f"  Fray:  {results['corners']['fray']:.2f}/1000")
        print(f"  Fill:  {results['corners']['fill']:.2f}/1000")
        print(f"  Angle: {results['corners']['angle']:.2f}/1000")

        print("\nEDGES:")
        print(f"  Fray: {results['edges']['fray']:.2f}/1000")
        print(f"  Fill: {results['edges']['fill']:.2f}/1000")

        print("\nCENTERING:")
        print("  Front:")
        fl = results['centering']['front_left']
        fr = results['centering']['front_right']
        ft = results['centering']['front_top']
        fb = results['centering']['front_bottom']
        print(f"    L/R: {fl:.1f}/{fr:.1f}")
        print(f"    T/B: {ft:.1f}/{fb:.1f}")

        print("  Back:")
        bl = results['centering']['back_left']
        br = results['centering']['back_right']
        bt = results['centering']['back_top']
        bb = results['centering']['back_bottom']
        print(f"    L/R: {bl:.1f}/{br:.1f}")
        print(f"    T/B: {bt:.1f}/{bb:.1f}")

        print("\n" + "="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test the card grading model')
    parser.add_argument('--front', type=str, required=True,
                       help='Path to front image of card')
    parser.add_argument('--back', type=str, required=True,
                       help='Path to back image of card')
    parser.add_argument('--device', type=str, default='mps',
                       help='Device to use (mps, cuda, or cpu)')

    args = parser.parse_args()

    # Check if image files exist
    if not Path(args.front).exists():
        print(f"Error: Front image not found at {args.front}")
        return

    if not Path(args.back).exists():
        print(f"Error: Back image not found at {args.back}")
        return

    # Initialize grader and make prediction
    grader = CardGrader(device=args.device)
    results = grader.predict(args.front, args.back)
    grader.print_results(results)


if __name__ == "__main__":
    main()

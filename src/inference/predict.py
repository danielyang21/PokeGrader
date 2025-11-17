"""
Inference script for predicting card grades.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class CardGradePredictor:
    """
    Wrapper class for making predictions on card images.
    """

    def __init__(self, model_path, device='cpu'):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
        """
        self.device = torch.device(device)

        # Load model
        from src.models.card_grader import CardGraderCNN
        self.model = CardGraderCNN(num_classes=10, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Grade mapping
        self.grade_names = {
            0: "Poor (1)",
            1: "Fair (2)",
            2: "Good (3)",
            3: "Very Good (4)",
            4: "Excellent (5)",
            5: "Excellent-Mint (6)",
            6: "Near Mint (7)",
            7: "Near Mint-Mint (8)",
            8: "Mint (9)",
            9: "Gem Mint (10)"
        }

    def predict_image(self, image_path):
        """
        Predict grade for a single image.

        Args:
            image_path: Path to card image

        Returns:
            Dictionary with predicted grade and confidence scores
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_grade = predicted_class.item() + 1  # Convert 0-9 to 1-10
        confidence_score = confidence.item()

        return {
            'predicted_grade': predicted_grade,
            'grade_name': self.grade_names[predicted_class.item()],
            'confidence': confidence_score,
            'all_probabilities': probabilities[0].cpu().numpy()
        }

    def predict_batch(self, image_paths):
        """
        Predict grades for multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            result['image_path'] = image_path
            results.append(result)
        return results

    def analyze_card_value(self, image_path, current_value, grading_cost=30):
        """
        Analyze whether grading is worth it based on predicted grade.

        Args:
            image_path: Path to card image
            current_value: Current ungraded value
            grading_cost: Cost of grading service

        Returns:
            Analysis dictionary
        """
        prediction = self.predict_image(image_path)
        predicted_grade = prediction['predicted_grade']

        # Rough multipliers based on typical PSA grading impact
        grade_multipliers = {
            1: 0.1, 2: 0.2, 3: 0.3, 4: 0.5, 5: 0.7,
            6: 1.0, 7: 1.5, 8: 2.5, 9: 5.0, 10: 10.0
        }

        multiplier = grade_multipliers.get(predicted_grade, 1.0)
        estimated_graded_value = current_value * multiplier
        net_gain = estimated_graded_value - current_value - grading_cost

        recommendation = "Worth grading" if net_gain > 0 else "Not worth grading"

        return {
            'predicted_grade': predicted_grade,
            'grade_name': prediction['grade_name'],
            'confidence': prediction['confidence'],
            'current_value': current_value,
            'estimated_graded_value': estimated_graded_value,
            'grading_cost': grading_cost,
            'net_gain': net_gain,
            'recommendation': recommendation
        }


def visualize_prediction(image_path, prediction):
    """
    Visualize image with prediction overlay.

    Args:
        image_path: Path to image
        prediction: Prediction dictionary
    """
    import matplotlib.pyplot as plt

    image = Image.open(image_path)

    plt.figure(figsize=(12, 6))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted Grade: {prediction['grade_name']}\n"
             f"Confidence: {prediction['confidence']:.2%}")
    plt.axis('off')

    # Show probability distribution
    plt.subplot(1, 2, 2)
    grades = [f"Grade {i+1}" for i in range(10)]
    probabilities = prediction['all_probabilities']

    plt.bar(grades, probabilities)
    plt.xlabel('Grade')
    plt.ylabel('Probability')
    plt.title('Grade Probability Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Card Grade Predictor")
    print("-" * 60)
    print("To use this predictor:")
    print("1. Train a model using src/training/train.py")
    print("2. Load the trained model:")
    print("   predictor = CardGradePredictor('checkpoints/best_model.pth')")
    print("3. Make predictions:")
    print("   result = predictor.predict_image('path/to/card.jpg')")
    print("   print(result)")

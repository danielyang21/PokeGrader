"""
Online/Streaming Training - Scrape and train on-the-fly
Scrapes cards, extracts features, trains, then discards images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import time
from io import BytesIO
from PIL import Image
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.card_grader import create_model
from scraper.enhanced_metadata_scraper import EnhancedMetadataScraper
from data.card_dataset import get_data_augmentation, GRADE_TO_IDX
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OnlineCardTrainer:
    """
    Trains models while scraping - doesn't save images locally.
    Features are extracted and used for training, then discarded.
    """

    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        load_pretrained: bool = True
    ):
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Initialize models
        self.corner_model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True).to(device)
        self.edge_model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True).to(device)
        self.centering_model = create_model(num_grade_classes=3, pretrained=True, freeze_backbone=True).to(device)

        # Add centering head (outputs 4 values: front_LR%, front_TB%, back_LR%, back_TB%)
        # We only predict LEFT and TOP percentages, RIGHT and BOTTOM are computed as 100 - LEFT/TOP
        self.centering_model.centering_head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # Changed from 8 to 4
            nn.Sigmoid()  # Added sigmoid to constrain to 0-1 range
        ).to(device)

        # Note: multiply by 100 to get percentages after prediction

        # Load pretrained models if available
        if load_pretrained:
            corner_path = Path("models/corner_model.pth")
            edge_path = Path("models/edge_model.pth")
            centering_path = Path("models/centering_model.pth")

            if corner_path.exists():
                logger.info(f"Loading pretrained corner model from {corner_path}")
                self.corner_model.load_state_dict(torch.load(corner_path, map_location=device))

            if edge_path.exists():
                logger.info(f"Loading pretrained edge model from {edge_path}")
                self.edge_model.load_state_dict(torch.load(edge_path, map_location=device))

            if centering_path.exists():
                logger.info(f"Loading pretrained centering model from {centering_path}")
                self.centering_model.load_state_dict(torch.load(centering_path, map_location=device))

        # Optimizers
        self.corner_optimizer = optim.Adam(self.corner_model.parameters(), lr=learning_rate)
        self.edge_optimizer = optim.Adam(self.edge_model.parameters(), lr=learning_rate)
        self.centering_optimizer = optim.Adam(self.centering_model.parameters(), lr=learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()

        # Transform for images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Stats
        self.cards_trained = 0
        self.corner_losses = []
        self.edge_losses = []
        self.centering_losses = []

    def extract_corner_images(self, front_img: Image.Image, back_img: Image.Image) -> torch.Tensor:
        """
        Extract 8 corner images (4 front + 4 back) from full card images.
        Returns tensor of shape [8, 3, 224, 224]
        """
        corner_size = 100  # pixels to extract from each corner
        corners = []

        for img in [front_img, back_img]:
            w, h = img.size

            # Top-left
            corners.append(self.transform(img.crop((0, 0, corner_size, corner_size))))
            # Top-right
            corners.append(self.transform(img.crop((w-corner_size, 0, w, corner_size))))
            # Bottom-left
            corners.append(self.transform(img.crop((0, h-corner_size, corner_size, h))))
            # Bottom-right
            corners.append(self.transform(img.crop((w-corner_size, h-corner_size, w, h))))

        return torch.stack(corners).to(self.device)  # [8, 3, 224, 224]

    def extract_edge_images(self, front_img: Image.Image, back_img: Image.Image) -> torch.Tensor:
        """
        Extract 8 edge images (4 front + 4 back) from full card images.
        Returns tensor of shape [8, 3, 224, 224]
        """
        edge_width = 50  # pixels to extract from each edge
        edges = []

        for img in [front_img, back_img]:
            w, h = img.size

            # Top edge
            edges.append(self.transform(img.crop((0, 0, w, edge_width))))
            # Right edge
            edges.append(self.transform(img.crop((w-edge_width, 0, w, h))))
            # Bottom edge
            edges.append(self.transform(img.crop((0, h-edge_width, w, h))))
            # Left edge
            edges.append(self.transform(img.crop((0, 0, edge_width, h))))

        return torch.stack(edges).to(self.device)  # [8, 3, 224, 224]

    def scrape_and_extract_features(self, cert_numbers: list) -> list:
        """
        Scrape cards and extract all features without saving images.

        Returns:
            List of dicts with features and metadata for each card
        """
        scraper = EnhancedMetadataScraper()
        card_data = []

        try:
            for i, cert_number in enumerate(tqdm(cert_numbers, desc="Scraping cards")):
                try:
                    # Extract metadata
                    metadata = scraper.extract_metadata(cert_number)

                    if not metadata:
                        logger.warning(f"Failed to get metadata for {cert_number}")
                        continue

                    # Get images from the driver (in memory)
                    try:
                        from selenium.webdriver.common.by import By

                        # Wait for images to load
                        time.sleep(2)

                        # Find all images on the page
                        images = scraper.driver.find_elements(By.TAG_NAME, "img")

                        # Look for front/back images (usually high resolution ones)
                        front_url = None
                        back_url = None

                        for img_elem in images:
                            src = img_elem.get_attribute('src')
                            alt = img_elem.get_attribute('alt') or ''

                            if src and ('front' in alt.lower() or 'front' in src.lower()):
                                front_url = src
                            elif src and ('back' in alt.lower() or 'back' in src.lower()):
                                back_url = src

                        # If not found by alt text, try to find high-res images
                        if not front_url or not back_url:
                            for img_elem in images:
                                src = img_elem.get_attribute('src')
                                if src and 'cloudfront' in src:  # TAG uses cloudfront for images
                                    if not front_url:
                                        front_url = src
                                    elif src != front_url:
                                        back_url = src
                                        break

                        if not front_url or not back_url:
                            logger.warning(f"Could not find images for {cert_number}")
                            continue

                        # Download images to memory
                        import requests
                        front_img = Image.open(BytesIO(requests.get(front_url).content)).convert('RGB')
                        back_img = Image.open(BytesIO(requests.get(back_url).content)).convert('RGB')

                        # Transform full card images
                        front_tensor = self.transform(front_img).unsqueeze(0).to(self.device)
                        back_tensor = self.transform(back_img).unsqueeze(0).to(self.device)

                        # Extract corner and edge images
                        corner_images = self.extract_corner_images(front_img, back_img)
                        edge_images = self.extract_edge_images(front_img, back_img)

                        card_data.append({
                            'cert_number': cert_number,
                            'front_image': front_tensor,
                            'back_image': back_tensor,
                            'corner_images': corner_images,  # [8, 3, 224, 224]
                            'edge_images': edge_images,      # [8, 3, 224, 224]
                            'metadata': metadata
                        })

                    except Exception as e:
                        logger.warning(f"Failed to load images for {cert_number}: {e}")
                        continue

                    # Rate limiting
                    if (i + 1) % 10 == 0:
                        time.sleep(2)

                except Exception as e:
                    logger.error(f"Error processing {cert_number}: {e}")
                    continue

        finally:
            scraper.close()

        logger.info(f"Successfully scraped {len(card_data)}/{len(cert_numbers)} cards")
        return card_data

    def train_on_batch(self, card_data: list):
        """Train models on a batch of cards."""

        # === TRAIN CORNER MODEL ===
        self.corner_model.train()
        corner_batch_loss = 0.0
        corner_count = 0

        for card in card_data:
            metadata = card['metadata']
            corners_meta = metadata.get('corners', [])

            if not corners_meta or len(corners_meta) != 8:
                continue

            # Get corner images [8, 3, 224, 224]
            corner_images = card['corner_images']

            for i, corner_meta in enumerate(corners_meta):
                fray = corner_meta.get('fray')
                fill = corner_meta.get('fill')
                angle = corner_meta.get('angle')

                if fray is None or fill is None:
                    continue

                # Create target [fray, fill, angle]
                if angle is None:  # Back corners don't have angle
                    angle = 1000  # Use perfect score for back corners

                target = torch.tensor([fray, fill, angle], dtype=torch.float32).unsqueeze(0).to(self.device)

                # Forward pass for this corner
                self.corner_optimizer.zero_grad()

                corner_img = corner_images[i:i+1]  # [1, 3, 224, 224]
                pred = self.corner_model.predict_corner(corner_img)

                loss = self.mse_loss(pred, target)
                loss.backward()
                self.corner_optimizer.step()

                corner_batch_loss += loss.item()
                corner_count += 1

        if corner_count > 0:
            avg_loss = corner_batch_loss / corner_count
            self.corner_losses.append(avg_loss)
            logger.info(f"Corner batch loss: {avg_loss:.4f}")

        # === TRAIN EDGE MODEL ===
        self.edge_model.train()
        edge_batch_loss = 0.0
        edge_count = 0

        for card in card_data:
            metadata = card['metadata']
            edges_meta = metadata.get('edges', [])

            if not edges_meta or len(edges_meta) != 8:
                continue

            # Get edge images [8, 3, 224, 224]
            edge_images = card['edge_images']

            for i, edge_meta in enumerate(edges_meta):
                fray = edge_meta.get('fray')
                fill = edge_meta.get('fill')

                if fray is None or fill is None:
                    continue

                # Create target [fray, fill]
                target = torch.tensor([fray, fill], dtype=torch.float32).unsqueeze(0).to(self.device)

                # Forward pass for this edge
                self.edge_optimizer.zero_grad()

                edge_img = edge_images[i:i+1]  # [1, 3, 224, 224]
                pred = self.edge_model.predict_edge(edge_img)

                loss = self.mse_loss(pred, target)
                loss.backward()
                self.edge_optimizer.step()

                edge_batch_loss += loss.item()
                edge_count += 1

        if edge_count > 0:
            avg_loss = edge_batch_loss / edge_count
            self.edge_losses.append(avg_loss)
            logger.info(f"Edge batch loss: {avg_loss:.4f}")

        # === TRAIN CENTERING MODEL ===
        self.centering_model.train()
        centering_batch_loss = 0.0
        centering_count = 0

        for card in card_data:
            metadata = card['metadata']

            # Check if centering data exists
            if not all([
                metadata.get('centering_front_lr'),
                metadata.get('centering_front_tb'),
                metadata.get('centering_back_lr'),
                metadata.get('centering_back_tb')
            ]):
                continue

            # Parse centering - only need LEFT and TOP percentages
            # RIGHT = 100 - LEFT, BOTTOM = 100 - TOP (computed in model output)
            def parse_centering_left_top(lr_str, tb_str):
                # "47L/53R" → extract 47 (left percentage)
                # "48T/52B" → extract 48 (top percentage)
                lr_parts = lr_str.replace('L', '').replace('R', '').split('/')
                tb_parts = tb_str.replace('T', '').replace('B', '').split('/')
                left_pct = float(lr_parts[0]) / 100.0  # Normalize to 0-1
                top_pct = float(tb_parts[0]) / 100.0   # Normalize to 0-1
                return [left_pct, top_pct]

            front_lt = parse_centering_left_top(
                metadata['centering_front_lr'],
                metadata['centering_front_tb']
            )
            back_lt = parse_centering_left_top(
                metadata['centering_back_lr'],
                metadata['centering_back_tb']
            )

            # Target is [front_L%, front_T%, back_L%, back_T%] in range 0-1
            target = torch.tensor(
                front_lt + back_lt,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            # Forward pass
            self.centering_optimizer.zero_grad()

            front_features = self.centering_model.extract_features(card['front_image'])
            back_features = self.centering_model.extract_features(card['back_image'])
            combined = torch.cat([front_features, back_features], dim=1)
            pred = self.centering_model.centering_head(combined)

            loss = self.mse_loss(pred, target)
            loss.backward()
            self.centering_optimizer.step()

            centering_batch_loss += loss.item()
            centering_count += 1

        if centering_count > 0:
            avg_loss = centering_batch_loss / centering_count
            self.centering_losses.append(avg_loss)
            logger.info(f"Centering batch loss: {avg_loss:.4f}")

        self.cards_trained += len(card_data)

        # Print summary
        logger.info(f"Batch complete - Trained on {len(card_data)} cards")
        logger.info(f"  Corners: {corner_count} samples")
        logger.info(f"  Edges: {edge_count} samples")
        logger.info(f"  Centering: {centering_count} samples")

    def train_online(
        self,
        cert_numbers: list,
        scrape_batch_size: int = 50,
        save_interval: int = 100
    ):
        """
        Main online training loop.
        Scrapes cards in batches, trains, discards images.

        Args:
            cert_numbers: List of certificate numbers to train on
            scrape_batch_size: How many cards to scrape before training
            save_interval: Save models every N cards
        """
        logger.info("="*60)
        logger.info("ONLINE TRAINING MODE")
        logger.info("="*60)
        logger.info(f"Total cards to process: {len(cert_numbers)}")
        logger.info(f"Scrape batch size: {scrape_batch_size}")

        # Process in batches
        for i in range(0, len(cert_numbers), scrape_batch_size):
            batch_certs = cert_numbers[i:i + scrape_batch_size]

            logger.info(f"\nProcessing batch {i//scrape_batch_size + 1}/{(len(cert_numbers) + scrape_batch_size - 1)//scrape_batch_size}")

            # Scrape and extract features
            card_data = self.scrape_and_extract_features(batch_certs)

            if not card_data:
                logger.warning("No valid cards in batch, skipping")
                continue

            # Train on this batch
            self.train_on_batch(card_data)

            # Clear memory
            del card_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Save checkpoints
            if self.cards_trained > 0 and self.cards_trained % save_interval == 0:
                self.save_models()

        # Final save
        self.save_models()

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total cards trained: {self.cards_trained}")
        if self.corner_losses:
            logger.info(f"Corner - Avg loss: {sum(self.corner_losses)/len(self.corner_losses):.4f}")
        if self.edge_losses:
            logger.info(f"Edge - Avg loss: {sum(self.edge_losses)/len(self.edge_losses):.4f}")
        if self.centering_losses:
            logger.info(f"Centering - Avg loss: {sum(self.centering_losses)/len(self.centering_losses):.4f}")

    def save_models(self):
        """Save model checkpoints."""
        Path("models").mkdir(exist_ok=True)

        torch.save(self.corner_model.state_dict(), "models/corner_model_online.pth")
        torch.save(self.edge_model.state_dict(), "models/edge_model_online.pth")
        torch.save(self.centering_model.state_dict(), "models/centering_model_online.pth")

        logger.info(f"✓ Saved models (after {self.cards_trained} cards)")


def main():
    """Main entry point for online training."""
    import argparse

    parser = argparse.ArgumentParser(description='Online training mode - scrape and train without saving images')
    parser.add_argument('--cert-file', type=str, default='data/cert_numbers.json',
                       help='JSON file with cert numbers (from cert_number_finder)')
    parser.add_argument('--num-cards', type=int, default=None,
                       help='Limit number of cards to train on (default: all)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Cards to scrape before training')
    parser.add_argument('--find-certs', action='store_true',
                       help='Run cert number finder first')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch without loading pretrained models')

    args = parser.parse_args()

    # Find cert numbers if requested
    if args.find_certs:
        logger.info("Running cert number finder...")
        from scraper.cert_number_finder import CertNumberFinder

        finder = CertNumberFinder(headless=True)
        finder.find_all_certs(
            start_url="https://my.taggrading.com/pop-report/Pokémon",
            max_depth=5,
            output_file=args.cert_file
        )

    # Load cert numbers
    cert_file = Path(args.cert_file)
    if not cert_file.exists():
        logger.error(f"Cert file not found: {args.cert_file}")
        logger.info("Run with --find-certs to generate cert numbers first")
        return

    with open(cert_file) as f:
        data = json.load(f)
        cert_numbers = data.get('cert_numbers', [])

    if not cert_numbers:
        logger.error("No cert numbers found in file")
        return

    logger.info(f"Loaded {len(cert_numbers)} cert numbers")

    # Limit if requested
    if args.num_cards:
        cert_numbers = cert_numbers[:args.num_cards]
        logger.info(f"Limited to {len(cert_numbers)} cards")

    # Start training
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    trainer = OnlineCardTrainer(device=device, load_pretrained=not args.no_pretrained)
    trainer.train_online(cert_numbers, scrape_batch_size=args.batch_size)


if __name__ == "__main__":
    main()

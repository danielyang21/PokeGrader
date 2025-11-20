"""
Enhanced metadata scraper for TAG cards.
Extracts detailed corner, edge, centering, and category scores.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from typing import Dict, Optional
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMetadataScraper:
    """Scraper for extracting detailed TAG card metadata."""

    def __init__(self):
        self.base_url = "https://my.taggrading.com"
        self.driver = None
        self.setup_driver()

    def setup_driver(self):
        """Setup Selenium WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')

        self.driver = webdriver.Chrome(options=chrome_options)
        logger.info("Chrome WebDriver initialized")

    def close(self):
        """Clean up."""
        if self.driver:
            self.driver.quit()

    def extract_metadata(self, cert_number: str) -> Optional[Dict]:
        """
        Extract detailed metadata for a card.

        Returns a dict with:
        - Corner scores (8 corners x 3 metrics each: fray, fill, angle)
        - Edge scores (8 edges x 2 metrics each: fray, fill)
        - Centering measurements (front/back L/R and T/B)
        - Category scores (Centering, Corners, Surface, Edges out of 250)
        - Total score out of 1000
        - Final grade (1-10)
        - DINGS counts
        """
        url = f"{self.base_url}/card/{cert_number}"
        logger.info(f"Extracting metadata for: {cert_number}")

        try:
            self.driver.get(url)
            time.sleep(5)  # Wait for page to load

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            text = soup.get_text()
            html = str(soup)

            metadata = {
                'cert_number': cert_number,
                'url': url,

                # Category scores (out of 250 each)
                'centering_score': None,
                'corners_score': None,
                'surface_score': None,
                'edges_score': None,
                'total_score': None,  # out of 1000
                'final_grade': None,  # 1-10 scale

                # DINGS counts
                'corners_front_dings': None,
                'corners_back_dings': None,
                'edges_front_dings': None,
                'edges_back_dings': None,
                'surface_front_dings': None,
                'surface_back_dings': None,

                # Centering measurements (string format like "49L/51R")
                'centering_front_lr': None,
                'centering_front_tb': None,
                'centering_back_lr': None,
                'centering_back_tb': None,


                # Corner scores (8 corners)
                'corners': [
                    {'position': 'front_top_left', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'front_top_right', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'front_bottom_left', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'front_bottom_right', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'back_top_left', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'back_top_right', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'back_bottom_left', 'fray': None, 'fill': None, 'angle': None},
                    {'position': 'back_bottom_right', 'fray': None, 'fill': None, 'angle': None},
                ],

                # Edge scores (8 edges)
                'edges': [
                    {'position': 'front_top', 'fray': None, 'fill': None},
                    {'position': 'front_right', 'fray': None, 'fill': None},
                    {'position': 'front_bottom', 'fray': None, 'fill': None},
                    {'position': 'front_left', 'fray': None, 'fill': None},
                    {'position': 'back_top', 'fray': None, 'fill': None},
                    {'position': 'back_right', 'fray': None, 'fill': None},
                    {'position': 'back_bottom', 'fray': None, 'fill': None},
                    {'position': 'back_left', 'fray': None, 'fill': None},
                ],

                # Dimensions
                'height': None,
                'width': None,
            }

            # Parse category scores - TAG's system has 4 categories worth 250 each
            logger.info("Parsing category scores...")
            # Note: TAG doesn't display individual category scores (centering/corners/edges/surface out of 250)
            # publicly on the card page. They only show the total TAG Score out of 1000.
            # We'll leave these as null unless we find a way to extract them.

            # Parse DINGS counts
            logger.info("\nParsing DINGS counts...")
            corners_dings = re.search(r'corners\s+F:\s*(\d+)\s*DINGS\s+B:\s*(\d+)\s*DINGS', text, re.IGNORECASE)
            if corners_dings:
                metadata['corners_front_dings'] = int(corners_dings.group(1))
                metadata['corners_back_dings'] = int(corners_dings.group(2))
                logger.info(f"  Corners - F: {corners_dings.group(1)} DINGS, B: {corners_dings.group(2)} DINGS")

            edges_dings = re.search(r'edges\s+F:\s*(\d+)\s*DINGS\s+B:\s*(\d+)\s*DINGS', text, re.IGNORECASE)
            if edges_dings:
                metadata['edges_front_dings'] = int(edges_dings.group(1))
                metadata['edges_back_dings'] = int(edges_dings.group(2))
                logger.info(f"  Edges - F: {edges_dings.group(1)} DINGS, B: {edges_dings.group(2)} DINGS")

            surface_dings = re.search(r'surface\s+F:\s*(\d+)\s*DINGS\s+B:\s*(\d+)\s*DINGS', text, re.IGNORECASE)
            if surface_dings:
                metadata['surface_front_dings'] = int(surface_dings.group(1))
                metadata['surface_back_dings'] = int(surface_dings.group(2))
                logger.info(f"  Surface - F: {surface_dings.group(1)} DINGS, B: {surface_dings.group(2)} DINGS")

            # Parse TAG Score (total score out of 1000)
            logger.info("\nParsing TAG Score...")
            # The TAG Score appears as a 3-4 digit number immediately before "TAG Score" in the text
            # Pattern in text: "988TAG Score" or "988 TAG Score"
            tag_score_match = re.search(r'(\d{3,4})\s*TAG Score', text, re.IGNORECASE)
            if tag_score_match:
                metadata['total_score'] = int(tag_score_match.group(1))
                logger.info(f"  TAG Score: {tag_score_match.group(1)}/1000")

            # Parse centering measurements (string format)
            logger.info("\nParsing centering measurements (string format)...")
            # Look for pattern like: centering F: 51L/49R 50T/50B B: 47L/53R 49T/51B
            centering = re.search(
                r'centering.*?F:\s*(\d+L/\d+R)\s+(\d+T/\d+B).*?B:\s*(\d+L/\d+R)\s+(\d+T/\d+B)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if centering:
                metadata['centering_front_lr'] = centering.group(1)
                metadata['centering_front_tb'] = centering.group(2)
                metadata['centering_back_lr'] = centering.group(3)
                metadata['centering_back_tb'] = centering.group(4)
                logger.info(f"  Front: {centering.group(1)} {centering.group(2)}")
                logger.info(f"  Back: {centering.group(3)} {centering.group(4)}")

            # Parse dimensions
            height_match = re.search(r'H:\s*([\d.]+)"', text)
            width_match = re.search(r'W:\s*([\d.]+)"', text)
            if height_match and width_match:
                metadata['height'] = float(height_match.group(1))
                metadata['width'] = float(width_match.group(1))
                logger.info(f"\nDimensions: {metadata['height']}\" x {metadata['width']}\"")

            # Parse final grade
            grade_match = re.search(r'(GEM MINT|MINT|NM-MT|NEAR MINT|EXCELLENT|VG-EX|VERY GOOD|GOOD|FAIR|POOR)', text, re.IGNORECASE)
            if grade_match:
                metadata['final_grade'] = grade_match.group(1)
                logger.info(f"Grade: {metadata['final_grade']}")

            # Extract individual corner scores
            # Patterns like "top LfrontFray:1000 Fill:1000 Angle:998"
            logger.info("\nExtracting individual corner scores...")

            # Front corners
            # Top Left Front
            tl_front = re.search(r'top\s*L\s*front\s*Fray:(\d+)\s*Fill:(\d+)\s*Angle:(\d+)', text, re.IGNORECASE)
            if tl_front:
                metadata['corners'][0]['fray'] = int(tl_front.group(1))
                metadata['corners'][0]['fill'] = int(tl_front.group(2))
                metadata['corners'][0]['angle'] = int(tl_front.group(3))
                logger.info(f"  Front TL: Fray={tl_front.group(1)}, Fill={tl_front.group(2)}, Angle={tl_front.group(3)}")

            # Top Right Front
            tr_front = re.search(r'top\s*R\s*front\s*Fray:(\d+)\s*Fill:(\d+)\s*Angle:(\d+)', text, re.IGNORECASE)
            if tr_front:
                metadata['corners'][1]['fray'] = int(tr_front.group(1))
                metadata['corners'][1]['fill'] = int(tr_front.group(2))
                metadata['corners'][1]['angle'] = int(tr_front.group(3))
                logger.info(f"  Front TR: Fray={tr_front.group(1)}, Fill={tr_front.group(2)}, Angle={tr_front.group(3)}")

            # Bottom Left Front
            bl_front = re.search(r'Bottom\s*L\s*front\s*Fray:(\d+)\s*Fill:(\d+)\s*Angle:(\d+)', text, re.IGNORECASE)
            if bl_front:
                metadata['corners'][2]['fray'] = int(bl_front.group(1))
                metadata['corners'][2]['fill'] = int(bl_front.group(2))
                metadata['corners'][2]['angle'] = int(bl_front.group(3))
                logger.info(f"  Front BL: Fray={bl_front.group(1)}, Fill={bl_front.group(2)}, Angle={bl_front.group(3)}")

            # Bottom Right Front
            br_front = re.search(r'Bottom\s*R\s*front\s*Fray:(\d+)\s*Fill:(\d+)\s*Angle:(\d+)', text, re.IGNORECASE)
            if br_front:
                metadata['corners'][3]['fray'] = int(br_front.group(1))
                metadata['corners'][3]['fill'] = int(br_front.group(2))
                metadata['corners'][3]['angle'] = int(br_front.group(3))
                logger.info(f"  Front BR: Fray={br_front.group(1)}, Fill={br_front.group(2)}, Angle={br_front.group(3)}")

            # Back corners
            # Top Left Back
            tl_back = re.search(r'top\s*L\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if tl_back:
                metadata['corners'][4]['fray'] = int(tl_back.group(1))
                metadata['corners'][4]['fill'] = int(tl_back.group(2))
                logger.info(f"  Back TL: Fray={tl_back.group(1)}, Fill={tl_back.group(2)}")

            # Top Right Back
            tr_back = re.search(r'top\s*R\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if tr_back:
                metadata['corners'][5]['fray'] = int(tr_back.group(1))
                metadata['corners'][5]['fill'] = int(tr_back.group(2))
                logger.info(f"  Back TR: Fray={tr_back.group(1)}, Fill={tr_back.group(2)}")

            # Bottom Left Back
            bl_back = re.search(r'Bottom\s*L\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if bl_back:
                metadata['corners'][6]['fray'] = int(bl_back.group(1))
                metadata['corners'][6]['fill'] = int(bl_back.group(2))
                logger.info(f"  Back BL: Fray={bl_back.group(1)}, Fill={bl_back.group(2)}")

            # Bottom Right Back
            br_back = re.search(r'Bottom\s*R\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if br_back:
                metadata['corners'][7]['fray'] = int(br_back.group(1))
                metadata['corners'][7]['fill'] = int(br_back.group(2))
                logger.info(f"  Back BR: Fray={br_back.group(1)}, Fill={br_back.group(2)}")

            # Extract individual edge scores
            # Patterns like "topfrontFray:1000 Fill:997" or "LeftfrontFray:1000 Fill:1000"
            logger.info("\nExtracting individual edge scores...")

            # Front edges
            top_front = re.search(r'top\s*front\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if top_front:
                metadata['edges'][0]['fray'] = int(top_front.group(1))
                metadata['edges'][0]['fill'] = int(top_front.group(2))
                logger.info(f"  Front Top: Fray={top_front.group(1)}, Fill={top_front.group(2)}")

            right_front = re.search(r'Right\s*front\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if right_front:
                metadata['edges'][1]['fray'] = int(right_front.group(1))
                metadata['edges'][1]['fill'] = int(right_front.group(2))
                logger.info(f"  Front Right: Fray={right_front.group(1)}, Fill={right_front.group(2)}")

            bottom_front = re.search(r'Bottom\s*front\s*Fray:(\d+)\s*Fill:(\d+)(?!\s*Angle)', text, re.IGNORECASE)
            if bottom_front:
                metadata['edges'][2]['fray'] = int(bottom_front.group(1))
                metadata['edges'][2]['fill'] = int(bottom_front.group(2))
                logger.info(f"  Front Bottom: Fray={bottom_front.group(1)}, Fill={bottom_front.group(2)}")

            left_front = re.search(r'Left\s*front\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if left_front:
                metadata['edges'][3]['fray'] = int(left_front.group(1))
                metadata['edges'][3]['fill'] = int(left_front.group(2))
                logger.info(f"  Front Left: Fray={left_front.group(1)}, Fill={left_front.group(2)}")

            # Back edges
            top_back = re.search(r'top\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if top_back:
                metadata['edges'][4]['fray'] = int(top_back.group(1))
                metadata['edges'][4]['fill'] = int(top_back.group(2))
                logger.info(f"  Back Top: Fray={top_back.group(1)}, Fill={top_back.group(2)}")

            right_back = re.search(r'Right\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if right_back:
                metadata['edges'][5]['fray'] = int(right_back.group(1))
                metadata['edges'][5]['fill'] = int(right_back.group(2))
                logger.info(f"  Back Right: Fray={right_back.group(1)}, Fill={right_back.group(2)}")

            bottom_back = re.search(r'Bottom\s*back\s*Fray:(\d+)\s*Fill:(\d+)(?!\s*Angle)', text, re.IGNORECASE)
            if bottom_back:
                metadata['edges'][6]['fray'] = int(bottom_back.group(1))
                metadata['edges'][6]['fill'] = int(bottom_back.group(2))
                logger.info(f"  Back Bottom: Fray={bottom_back.group(1)}, Fill={bottom_back.group(2)}")

            left_back = re.search(r'Left\s*back\s*Fray:(\d+)\s*Fill:(\d+)', text, re.IGNORECASE)
            if left_back:
                metadata['edges'][7]['fray'] = int(left_back.group(1))
                metadata['edges'][7]['fill'] = int(left_back.group(2))
                logger.info(f"  Back Left: Fray={left_back.group(1)}, Fill={left_back.group(2)}")

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Test with one card."""
    scraper = EnhancedMetadataScraper()

    try:
        # Test with C3565664 (a card we know exists)
        logger.info("="*60)
        logger.info("Testing enhanced metadata scraper")
        logger.info("="*60)

        metadata = scraper.extract_metadata("C3565664")

        if metadata:
            logger.info("\n\nSUCCESS! Extracted metadata:")
            print(json.dumps(metadata, indent=2))
        else:
            logger.error("FAILED to extract metadata")

    finally:
        scraper.close()


if __name__ == "__main__":
    main()

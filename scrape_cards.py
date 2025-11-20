#!/usr/bin/env python3
"""
Consolidated TAG Grading Card Scraper
Scrapes card grading data from TAG Grading website for training purposes.
"""

from src.scraper.enhanced_metadata_scraper import EnhancedMetadataScraper
import json
from pathlib import Path
import logging
import time
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scrape_cards(cert_numbers, output_dir="src/scraper/data/tag_cards"):
    """
    Scrape card data for given cert numbers.

    Args:
        cert_numbers: List of certificate numbers to scrape
        output_dir: Directory to save scraped data
    """
    scraper = EnhancedMetadataScraper()
    output_path = Path(output_dir)

    success_count = 0
    failed_cards = []

    try:
        for i, cert_number in enumerate(cert_numbers, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(cert_numbers)}] Processing {cert_number}...")
            logger.info(f"{'='*60}")

            try:
                metadata = scraper.extract_metadata(cert_number)

                if metadata:
                    # Save to file
                    card_dir = output_path / cert_number
                    card_dir.mkdir(parents=True, exist_ok=True)
                    output_file = card_dir / "card_data.json"

                    with open(output_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"  ✓ Saved to {output_file}")
                    logger.info(f"  Grade: {metadata.get('final_grade')}")
                    logger.info(f"  TAG Score: {metadata.get('total_score', 'N/A')}")
                    success_count += 1
                else:
                    logger.error(f"  ✗ Failed to extract metadata")
                    failed_cards.append(cert_number)

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                failed_cards.append(cert_number)

            # Rate limiting
            if i < len(cert_numbers):
                time.sleep(2)

    finally:
        scraper.close()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SCRAPING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total cards: {len(cert_numbers)}")
    logger.info(f"Successfully scraped: {success_count}")
    logger.info(f"Failed: {len(failed_cards)}")

    if failed_cards:
        logger.info(f"\nFailed cards: {', '.join(failed_cards)}")

    return success_count, failed_cards


def get_all_cert_numbers(data_dir="src/scraper/data/tag_cards"):
    """Get all cert numbers from existing card directories."""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    cert_numbers = []
    for card_dir in sorted(base_dir.iterdir()):
        if card_dir.is_dir():
            cert_numbers.append(card_dir.name)

    return cert_numbers


def main():
    parser = argparse.ArgumentParser(description='Scrape TAG Grading card data')
    parser.add_argument('--cert-numbers', '-c', nargs='+',
                       help='Specific cert numbers to scrape')
    parser.add_argument('--rescrape-all', '-a', action='store_true',
                       help='Re-scrape all existing cards')
    parser.add_argument('--output-dir', '-o', default='src/scraper/data/tag_cards',
                       help='Output directory for scraped data')

    args = parser.parse_args()

    if args.cert_numbers:
        cert_numbers = args.cert_numbers
        logger.info(f"Scraping {len(cert_numbers)} specified cards...")
    elif args.rescrape_all:
        cert_numbers = get_all_cert_numbers(args.output_dir)
        logger.info(f"Re-scraping all {len(cert_numbers)} existing cards...")
    else:
        logger.error("Please specify cert numbers with -c or use --rescrape-all")
        parser.print_help()
        return

    if not cert_numbers:
        logger.error("No cards to scrape")
        return

    scrape_cards(cert_numbers, args.output_dir)


if __name__ == "__main__":
    main()

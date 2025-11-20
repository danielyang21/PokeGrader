#!/usr/bin/env python3
"""
Prepare training data from scraped TAG cards
Converts JSON files into format suitable for model training
"""

import json
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_card_data(data_dir="src/scraper/data/tag_cards"):
    """Load all card data from JSON files."""
    base_dir = Path(data_dir)
    cards = []

    for card_dir in sorted(base_dir.iterdir()):
        if card_dir.is_dir():
            json_file = card_dir / "card_data.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                    cards.append(data)

    logger.info(f"Loaded {len(cards)} cards")
    return cards


def convert_to_dataframe(cards):
    """Convert card data to pandas DataFrame for training."""
    rows = []

    for card in cards:
        row = {
            'cert_number': card['cert_number'],
            'final_grade': card['final_grade'],
            'total_score': card['total_score'],

            # Centering (string format)
            'centering_front_lr': card['centering_front_lr'],
            'centering_front_tb': card['centering_front_tb'],
            'centering_back_lr': card['centering_back_lr'],
            'centering_back_tb': card['centering_back_tb'],

            # Dimensions
            'height': card['height'],
            'width': card['width'],
        }

        # Add corner data (8 corners)
        for corner in card['corners']:
            pos = corner['position']
            row[f'{pos}_fray'] = corner['fray']
            row[f'{pos}_fill'] = corner['fill']
            row[f'{pos}_angle'] = corner['angle']

        # Add edge data (8 edges)
        for edge in card['edges']:
            pos = edge['position']
            row[f'{pos}_fray'] = edge['fray']
            row[f'{pos}_fill'] = edge['fill']

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def analyze_data(df):
    """Analyze the dataset."""
    logger.info("\n" + "="*60)
    logger.info("DATASET ANALYSIS")
    logger.info("="*60)

    logger.info(f"\nTotal cards: {len(df)}")

    logger.info(f"\nGrade distribution:")
    grade_counts = df['final_grade'].value_counts()
    for grade, count in grade_counts.items():
        logger.info(f"  {grade}: {count}")

    logger.info(f"\nCards with TAG Score: {df['total_score'].notna().sum()}")
    logger.info(f"Cards without TAG Score: {df['total_score'].isna().sum()}")

    logger.info(f"\nCentering data completeness:")
    logger.info(f"  Front L/R: {df['centering_front_lr'].notna().sum()}/{len(df)}")
    logger.info(f"  Front T/B: {df['centering_front_tb'].notna().sum()}/{len(df)}")
    logger.info(f"  Back L/R: {df['centering_back_lr'].notna().sum()}/{len(df)}")
    logger.info(f"  Back T/B: {df['centering_back_tb'].notna().sum()}/{len(df)}")

    # Check corner/edge data completeness
    corner_cols = [col for col in df.columns if 'corner' in col or ('front_' in col and 'fray' in col)]
    complete_corners = df[corner_cols].notna().all(axis=1).sum()
    logger.info(f"\nCards with complete corner data: {complete_corners}/{len(df)}")

    edge_cols = [col for col in df.columns if 'edge' in col or ('_fray' in col and 'fill' in col.replace('_fray', '_fill'))]
    complete_edges = df[edge_cols].notna().all(axis=1).sum()
    logger.info(f"Cards with complete edge data: {complete_edges}/{len(df)}")

    return df


def main():
    # Load data
    cards = load_all_card_data()

    if not cards:
        logger.error("No card data found!")
        return

    # Convert to DataFrame
    df = convert_to_dataframe(cards)

    # Analyze
    analyze_data(df)

    # Save to CSV for training
    output_file = "data/training_data.csv"
    Path("data").mkdir(exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved training data to {output_file}")

    # Also save summary stats
    summary_file = "data/data_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("TAG Grading Training Data Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total cards: {len(df)}\n\n")
        f.write("Grade distribution:\n")
        for grade, count in df['final_grade'].value_counts().items():
            f.write(f"  {grade}: {count}\n")
        f.write(f"\nFeatures: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n")

    logger.info(f"✓ Saved summary to {summary_file}")


if __name__ == "__main__":
    main()

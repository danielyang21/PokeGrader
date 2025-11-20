"""
Extract current cert numbers from the running scraper log.
Parses the log file to find all unique cert numbers discovered so far.
"""

import re
import json
from pathlib import Path

def extract_certs_from_log(log_file: str = "slow_scraper.log") -> set:
    """Extract all cert numbers mentioned in the log file."""
    certs = set()

    # Pattern to match cert numbers (Letter + 7-8 digits)
    cert_pattern = re.compile(r'\b([A-Z]\d{7,8})\b')

    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for lines that mention finding certs
                if 'Found cert:' in line or 'cert' in line.lower():
                    matches = cert_pattern.findall(line)
                    certs.update(matches)
    except FileNotFoundError:
        print(f"Log file {log_file} not found")
        return set()

    return certs


def main():
    print("Extracting certs from slow_scraper.log...")
    certs = extract_certs_from_log("slow_scraper.log")

    if not certs:
        print("No cert numbers found in log file")
        print("Note: The scraper may not be logging individual certs.")
        print("The cert count is stored in memory and only written at the end.")
        return

    # Save to file
    cert_list = sorted(list(certs))
    output = {
        'total': len(cert_list),
        'cert_numbers': cert_list
    }

    output_file = 'data/certs_extracted_from_log.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExtracted {len(cert_list)} unique cert numbers")
    print(f"Saved to: {output_file}")
    print(f"\nSample (first 10): {cert_list[:10]}")


if __name__ == "__main__":
    main()

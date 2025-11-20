"""
Scrape certificate numbers from TAG's population report
Navigates through the pop report hierarchy to find all graded cards
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
import json
from urllib.parse import urljoin, quote, unquote

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CertNumberFinder:
    """Find certificate numbers from TAG population report."""

    def __init__(self, headless: bool = True, output_file: str = None, autosave_interval: int = 10):
        """Initialize the cert number finder."""
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=options)
        self.base_url = "https://my.taggrading.com"
        self.found_certs = set()
        self.output_file = output_file
        self.autosave_interval = autosave_interval
        self.certs_since_save = 0

    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()

    def extract_cert_numbers_from_page(self, url: str) -> list:
        """
        Extract certificate numbers from a specific pop report page.

        Args:
            url: URL of the pop report page

        Returns:
            List of certificate numbers found
        """
        logger.info(f"Extracting cert numbers from: {url}")

        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Look for cert numbers in the page
            # They appear as links or text in the format: E8163336, H2851725, N2607372
            cert_numbers = []

            # Try to find cert number links
            links = soup.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)

                # Check if this looks like a cert number link
                if '/card/' in href:
                    # Extract cert number from URL: /card/E8163336
                    cert = href.split('/card/')[-1].split('?')[0].split('/')[0]
                    if cert and len(cert) >= 7:  # Cert numbers are typically 8+ chars
                        cert_numbers.append(cert)
                        logger.debug(f"  Found cert: {cert}")

                # Also check link text for cert patterns
                if text and len(text) >= 7 and any(text.startswith(prefix) for prefix in ['E', 'H', 'N', 'A', 'B']):
                    cert_numbers.append(text)

            # Also search for cert numbers in table cells or divs
            for elem in soup.find_all(['td', 'div', 'span']):
                text = elem.get_text(strip=True)
                # Match pattern: Letter followed by 7+ digits
                if text and len(text) >= 7 and any(text.startswith(prefix) for prefix in ['E', 'H', 'N', 'A', 'B']):
                    if text[1:].replace(' ', '').isdigit() or text[1:8].isdigit():
                        cert_numbers.append(text.replace(' ', ''))

            unique_certs = list(set(cert_numbers))
            logger.info(f"  Found {len(unique_certs)} cert numbers")

            return unique_certs

        except Exception as e:
            logger.error(f"Error extracting cert numbers: {e}")
            return []

    def get_sub_links(self, url: str) -> list:
        """
        Get all sub-links from a pop report page.
        These lead to deeper levels (years -> sets -> cards -> variations).

        Args:
            url: URL of current pop report page

        Returns:
            List of sub-page URLs
        """
        try:
            self.driver.get(url)
            time.sleep(2)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            sub_links = []

            # Find all links that continue into pop report
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')

                # Pop report links contain /pop-report/
                if '/pop-report/' in href and href != url:
                    full_url = urljoin(self.base_url, href)

                    # Avoid duplicates and circular links
                    if full_url not in sub_links and full_url != url:
                        sub_links.append(full_url)

            logger.info(f"Found {len(sub_links)} sub-links from {url}")
            return sub_links

        except Exception as e:
            logger.error(f"Error getting sub-links: {e}")
            return []

    def scrape_pop_report_recursive(
        self,
        start_url: str,
        max_depth: int = 5,
        current_depth: int = 0,
        visited: set = None
    ) -> set:
        """
        Recursively scrape the pop report, finding all cert numbers.

        Args:
            start_url: Starting URL
            max_depth: Maximum recursion depth
            current_depth: Current depth in recursion
            visited: Set of visited URLs

        Returns:
            Set of all cert numbers found
        """
        if visited is None:
            visited = set()

        if current_depth >= max_depth or start_url in visited:
            return self.found_certs

        visited.add(start_url)
        logger.info(f"\n{'  ' * current_depth}Depth {current_depth}: {start_url}")

        # Extract cert numbers from this page
        certs = self.extract_cert_numbers_from_page(start_url)
        new_certs = len(certs) - len(self.found_certs & set(certs))
        self.found_certs.update(certs)

        logger.info(f"{'  ' * current_depth}Total certs found so far: {len(self.found_certs)}")

        # Auto-save every N new certs
        if new_certs > 0:
            self.certs_since_save += new_certs
            if self.certs_since_save >= self.autosave_interval and self.output_file:
                self.save_cert_numbers(self.output_file)
                self.certs_since_save = 0

        # Get sub-links and recurse
        if current_depth < max_depth:
            sub_links = self.get_sub_links(start_url)

            for sub_link in sub_links:
                if sub_link not in visited:
                    time.sleep(1)  # Rate limiting
                    self.scrape_pop_report_recursive(
                        sub_link,
                        max_depth,
                        current_depth + 1,
                        visited
                    )

        return self.found_certs

    def save_cert_numbers(self, output_file: str = "cert_numbers.json"):
        """Save found cert numbers to file."""
        cert_list = sorted(list(self.found_certs))

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'total': len(cert_list),
                'cert_numbers': cert_list
            }, f, indent=2)

        logger.info(f"\n✓ Saved {len(cert_list)} cert numbers to {output_file}")

    def find_all_certs(
        self,
        start_url: str = "https://my.taggrading.com/pop-report/Pokémon",
        max_depth: int = 5,
        output_file: str = "data/cert_numbers.json"
    ):
        """
        Main function to find all certificate numbers.

        Args:
            start_url: Starting URL for pop report
            max_depth: How deep to recurse
            output_file: Where to save cert numbers
        """
        logger.info("="*60)
        logger.info("CERT NUMBER FINDER")
        logger.info("="*60)
        logger.info(f"Starting URL: {start_url}")
        logger.info(f"Max depth: {max_depth}")

        try:
            self.scrape_pop_report_recursive(start_url, max_depth=max_depth)
            self.save_cert_numbers(output_file)

            # Print summary
            logger.info("\n" + "="*60)
            logger.info("SUMMARY")
            logger.info("="*60)
            logger.info(f"Total cert numbers found: {len(self.found_certs)}")

            # Show a sample
            sample = list(self.found_certs)[:10]
            logger.info(f"\nSample cert numbers:")
            for cert in sample:
                logger.info(f"  {cert}")

        finally:
            self.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Find cert numbers from TAG pop report')
    parser.add_argument('--start-url', type=str,
                       default='https://my.taggrading.com/pop-report/Pokémon',
                       help='Starting URL')
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Maximum recursion depth')
    parser.add_argument('--output', type=str, default='data/cert_numbers.json',
                       help='Output file for cert numbers')
    parser.add_argument('--no-headless', action='store_true',
                       help='Run with visible browser')
    parser.add_argument('--autosave-interval', type=int, default=10,
                       help='Save every N new certs found (default: 10)')

    args = parser.parse_args()

    finder = CertNumberFinder(
        headless=not args.no_headless,
        output_file=args.output,
        autosave_interval=args.autosave_interval
    )
    finder.find_all_certs(
        start_url=args.start_url,
        max_depth=args.max_depth,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

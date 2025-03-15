import argparse
import asyncio
import aiohttp
import aiofiles
import os
import random
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class FontScraper:
    """A class to scrape fonts from 1001fonts.com using asyncio"""
    
    def __init__(self, output_dir="fonts", delay=2, max_pages=None, concurrency=10):
        """
        Initialize the scraper
        
        Args:
            output_dir: Directory to save fonts
            delay: Base delay between requests in seconds
            max_pages: Maximum number of pages to scrape (None for all)
            concurrency: Maximum number of concurrent downloads
        """
        self.base_url = "https://www.1001fonts.com"
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.semaphore = None  # Will be initialized in scrape_fonts
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    async def scrape_fonts(self, url_path="/free-for-commercial-use-fonts.html"):
        """
        Scrape fonts from the given URL path
        
        Args:
            url_path: Path to scrape fonts from
        """
        # Initialize semaphore for limiting concurrent downloads
        self.semaphore = asyncio.Semaphore(self.concurrency)
        
        page = 1
        has_next_page = True
        
        # Create a single session for all requests
        async with aiohttp.ClientSession(headers=self.headers) as session:
            while has_next_page:
                if self.max_pages and page > self.max_pages:
                    print(f"Reached maximum pages limit ({self.max_pages})")
                    break
                    
                current_url = f"{self.base_url}{url_path}"
                if page > 1:
                    current_url += f"?page={page}"
                    
                print(f"Scraping page {page} from {current_url}")
                
                try:
                    # Fetch the page
                    async with session.get(current_url) as response:
                        if response.status != 200:
                            print(f"Error: Got status code {response.status}")
                            break
                            
                        html = await response.text()
                        
                    # Parse the HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find all font items on the page
                    font_items = soup.select('li.font-list-item')
                    
                    if not font_items:
                        print("No fonts found on this page")
                        has_next_page = False
                        continue
                    
                    print(f"Found {len(font_items)} fonts on page {page}")
                    
                    # Extract font data from all items
                    font_data_list = []
                    for item in font_items:
                        font_data = self.extract_font_data(item)
                        if font_data:
                            font_data_list.append(font_data)
                    
                    # Download fonts concurrently
                    if font_data_list:
                        tasks = [
                            self.download_font(session, font_data) 
                            for font_data in font_data_list
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Count successful and failed downloads
                        successful = sum(1 for r in results if r is True)
                        failed = len(results) - successful
                        print(f"Downloaded {successful} fonts, {failed} failed")
                    
                    # Check if there's a next page
                    next_page = soup.select_one('li.next:not(.disabled)')
                    has_next_page = next_page is not None
                    
                    page += 1
                    
                    # Random delay between page requests
                    sleep_time = self.delay + random.uniform(0.5, 1.5)
                    print(f"Sleeping for {sleep_time:.2f} seconds before fetching next page")
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"Error fetching page {page}: {e}")
                    break
                    
            print("Scraping completed!")
    
    def extract_font_data(self, item):
        """
        Extract font data from a BeautifulSoup item
        
        Args:
            item: BeautifulSoup element containing font information
            
        Returns:
            dict: Dictionary with font data or None if extraction failed
        """
        try:
            # Extract font name
            name_elem = item.select_one('h6')
            if not name_elem:
                print("Could not find font name, skipping")
                return None
                
            # Get just the main font name without the style count or author
            font_name = name_elem.get_text().strip().split(' +')[0].strip()
            
            # Extract author name if available
            author_elem = item.select_one('small.text-muted a')
            author = author_elem.get_text().strip() if author_elem else "Unknown"
            
            # Extract download link
            download_elem = item.select_one('a.btn.btn-success')
            if not download_elem:
                print(f"No download link found for {font_name}, skipping")
                return None
                
            download_path = download_elem['href']
            download_url = urljoin(self.base_url, download_path)
            
            # Format font filename (remove special characters)
            safe_font_name = re.sub(r'[^\w\-\.]', '_', font_name)
            filename = f"{safe_font_name}.zip"
            filepath = os.path.join(self.output_dir, filename)
            
            return {
                'name': font_name,
                'author': author,
                'download_url': download_url,
                'filename': filename,
                'filepath': filepath
            }
            
        except Exception as e:
            print(f"Error extracting font data: {e}")
            return None
    
    async def download_font(self, session, font_data):
        """
        Download a single font asynchronously
        
        Args:
            session: aiohttp ClientSession to use for requests
            font_data: Dictionary containing font information
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Use semaphore to limit concurrent downloads
        async with self.semaphore:
            try:
                font_name = font_data['name']
                author = font_data['author']
                download_url = font_data['download_url']
                filepath = font_data['filepath']
                
                # Skip if already downloaded
                if os.path.exists(filepath):
                    print(f"Font {font_name} already downloaded, skipping")
                    return True
                    
                print(f"Downloading {font_name} by {author}")
                
                # Add a small random delay to avoid hammering the server
                await asyncio.sleep(random.uniform(0.2, 0.5))
                
                # Download the font
                async with session.get(download_url) as response:
                    if response.status != 200:
                        print(f"Error downloading {font_name}: Status {response.status}")
                        return False
                        
                    # Save the font file
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(await response.read())
                    
                print(f"Saved {font_name} to {filepath}")
                return True
                
            except Exception as e:
                print(f"Error downloading {font_data.get('name', 'unknown font')}: {e}")
                return False


async def main_async():
    """Async main function to run the scraper"""
    parser = argparse.ArgumentParser(description="Scrape fonts from 1001fonts.com")
    parser.add_argument("--output", "-o", default="fonts", help="Output directory")
    parser.add_argument("--delay", "-d", type=float, default=2.0, help="Base delay between requests in seconds")
    parser.add_argument("--pages", "-p", type=int, default=None, help="Maximum number of pages to scrape")
    parser.add_argument("--category", "-c", default="/free-for-commercial-use-fonts.html", 
                        help="Font category path (default: free for commercial use)")
    parser.add_argument("--concurrency", "-n", type=int, default=10, 
                        help="Maximum number of concurrent downloads")
    
    args = parser.parse_args()
    
    print("1001Fonts.com Scraper (Async)")
    print("---------------------")
    print(f"Output directory: {args.output}")
    print(f"Base delay: {args.delay} seconds")
    print(f"Max pages: {'All' if args.pages is None else args.pages}")
    print(f"Category: {args.category}")
    print(f"Concurrency: {args.concurrency}")
    print("---------------------")
    
    scraper = FontScraper(
        output_dir=args.output,
        delay=args.delay,
        max_pages=args.pages,
        concurrency=args.concurrency
    )
    
    await scraper.scrape_fonts(args.category)


def main():
    """Entry point for the script"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
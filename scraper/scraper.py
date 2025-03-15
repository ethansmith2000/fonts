import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import os
import random
import argparse
import re
import time
from urllib.parse import urljoin


class FontScraper:
    """A class to scrape fonts from 1001fonts.com using asyncio"""
    
    def __init__(self, output_dir="fonts", delay=1.0, max_pages=None, 
                 concurrency=5, respect_robots=True):
        """
        Initialize the scraper
        
        Args:
            output_dir: Directory to save fonts
            delay: Base delay between requests in seconds
            max_pages: Maximum number of pages to scrape (None for all)
            concurrency: Maximum number of concurrent downloads
            respect_robots: Whether to check and respect robots.txt
        """
        self.base_url = "https://www.1001fonts.com"
        self.output_dir = output_dir
        self.delay = delay
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.respect_robots = respect_robots
        self.download_count = 0
        self.request_count = 0
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    async def _check_robots_txt(self, session):
        """Check robots.txt for allowed crawling"""
        try:
            async with session.get(f"{self.base_url}/robots.txt") as response:
                if response.status == 200:
                    robots_content = await response.text()
                    print("Checking robots.txt...")
                    
                    # Basic check for disallow rules that might affect our scraping
                    disallowed = []
                    for line in robots_content.splitlines():
                        if line.lower().startswith('disallow:'):
                            path = line.split(':', 1)[1].strip()
                            disallowed.append(path)
                    
                    print(f"Found {len(disallowed)} disallow rules in robots.txt")
                    print("Consider respecting these restrictions")
        except Exception as e:
            print(f"Could not check robots.txt: {e}")
    
    async def _delay_request(self):
        """Apply a random delay between requests with jitter"""
        self.request_count += 1
        
        # After every 20 requests, add an extra delay to avoid patterns
        extra = 2.0 if self.request_count % 20 == 0 else 0
        
        # Base delay plus jitter plus potential extra delay
        sleep_time = self.delay + random.uniform(0, 0.5) + extra
        await asyncio.sleep(sleep_time)
    
    async def scrape_fonts(self, url_path="/free-for-commercial-use-fonts.html"):
        """
        Scrape fonts from the given URL path
        
        Args:
            url_path: Path to scrape fonts from
        """
        page = 1
        has_next_page = True
        font_details = []  # Store font details for concurrent downloading
        
        print(f"Starting to scrape {url_path} with concurrency level {self.concurrency}")
        print(f"Base delay between requests: {self.delay} seconds")
        start_time = time.time()
        
        # Create a semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # Create a single session for all requests
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Check robots.txt if enabled
            if self.respect_robots:
                await self._check_robots_txt(session)
            
            while has_next_page:
                if self.max_pages and page > self.max_pages:
                    print(f"Reached maximum pages limit ({self.max_pages})")
                    break
                    
                current_url = f"{self.base_url}{url_path}"
                if page > 1:
                    current_url += f"?page={page}"
                    
                print(f"Scraping page {page} from {current_url}")
                
                try:
                    # Apply delay before request
                    await self._delay_request()
                    
                    # Fetch the page
                    async with session.get(current_url) as response:
                        # Check for rate limiting
                        if response.status == 429:
                            print("Rate limited! Waiting 60 seconds before retrying...")
                            await asyncio.sleep(60)
                            continue
                            
                        if response.status != 200:
                            print(f"Error: Got status code {response.status}")
                            
                            # Exponential backoff on error
                            backoff_time = min(30, 2 ** (page % 5))
                            print(f"Backing off for {backoff_time} seconds...")
                            await asyncio.sleep(backoff_time)
                            continue
                            
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
                    
                    # Extract font details from each item
                    page_font_details = []
                    for item in font_items:
                        font_detail = self._extract_font_details(item)
                        if font_detail:
                            page_font_details.append(font_detail)
                    
                    if page_font_details:
                        # Create tasks for downloading fonts from this page
                        tasks = [
                            self._download_font(session, semaphore, font_detail) 
                            for font_detail in page_font_details
                        ]
                        
                        # Start downloads for this page concurrently
                        # We'll gather them all at the end
                        for task in tasks:
                            asyncio.create_task(task)
                    
                    # Check if there's a next page
                    next_page = soup.select_one('li.next:not(.disabled)')
                    has_next_page = next_page is not None
                    
                    page += 1
                    
                except Exception as e:
                    print(f"Error fetching page {page}: {e}")
                    # Exponential backoff on error
                    backoff_time = min(30, 2 ** (page % 5))
                    print(f"Backing off for {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                    continue
            
            # Wait for all pending download tasks to complete
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            await asyncio.gather(*pending, return_exceptions=True)
        
        elapsed = time.time() - start_time
        print(f"Scraping completed! Downloaded {self.download_count} fonts in {elapsed:.2f} seconds")
    
    def _extract_font_details(self, item):
        """
        Extract font details from a font list item
        
        Args:
            item: BeautifulSoup element containing font information
            
        Returns:
            Dict with font details or None if extraction fails
        """
        try:
            # Extract font name
            name_elem = item.select_one('h6')
            if not name_elem:
                return None
                
            # Get just the main font name without the style count or author
            font_name = name_elem.get_text().strip().split(' +')[0].strip()
            
            # Extract author name if available
            author_elem = item.select_one('small.text-muted a')
            author = author_elem.get_text().strip() if author_elem else "Unknown"
            
            # Extract download link
            download_elem = item.select_one('a.btn.btn-success')
            if not download_elem:
                return None
                
            download_path = download_elem['href']
            download_url = urljoin(self.base_url, download_path)
            
            # Format font filename (remove special characters)
            safe_font_name = re.sub(r'[^\w\-\.]', '_', font_name)
            
            return {
                'name': font_name,
                'author': author,
                'download_url': download_url,
                'safe_name': safe_font_name
            }
            
        except Exception as e:
            print(f"Error extracting font details: {e}")
            return None
    
    async def _download_font(self, session, semaphore, font_detail):
        """
        Download a font based on its details
        
        Args:
            session: aiohttp ClientSession
            semaphore: Asyncio semaphore for limiting concurrency
            font_detail: Dict containing font details
        """
        # Use semaphore to limit concurrent downloads
        async with semaphore:
            try:
                filename = f"{font_detail['safe_name']}.zip"
                filepath = os.path.join(self.output_dir, filename)
                
                # Skip if already downloaded
                if os.path.exists(filepath):
                    print(f"Font {font_detail['name']} already downloaded, skipping")
                    return
                    
                print(f"Downloading {font_detail['name']} by {font_detail['author']}")
                
                # Apply delay before request
                await self._delay_request()
                
                # Download the font
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        async with session.get(font_detail['download_url']) as download_response:
                            # Check for rate limiting
                            if download_response.status == 429:
                                print(f"Rate limited downloading {font_detail['name']}! Waiting before retry...")
                                await asyncio.sleep(30 + random.uniform(0, 30))  # Randomized backoff
                                retry_count += 1
                                continue
                                
                            if download_response.status != 200:
                                print(f"Error downloading {font_detail['name']}: Status {download_response.status}")
                                retry_count += 1
                                await asyncio.sleep(5 * retry_count)  # Increasing backoff
                                continue
                            
                            # Save the font file
                            async with aiofiles.open(filepath, 'wb') as f:
                                await f.write(await download_response.read())
                            
                            # Successfully downloaded
                            self.download_count += 1
                            print(f"Saved {font_detail['name']} to {filepath}")
                            break
                            
                    except Exception as e:
                        print(f"Error downloading {font_detail['name']}: {e}")
                        retry_count += 1
                        await asyncio.sleep(5 * retry_count)  # Increasing backoff
                
                if retry_count >= max_retries:
                    print(f"Failed to download {font_detail['name']} after {max_retries} retries")
                
            except Exception as e:
                print(f"Error processing {font_detail.get('name', 'unknown font')}: {e}")


async def main_async():
    """Async main function to run the scraper"""
    
    # Popular Style Categories:
    # Script/Handwritten: /script-fonts.html, /handwritten-fonts.html, /calligraphy-fonts.html
    # Decorative: /fancy-fonts.html, /elegant-fonts.html, /3d-fonts.html, /outlined-fonts.html
    # Themed: /grunge-fonts.html, /vintage-fonts.html, /retro-fonts.html, /futuristic-fonts.html
    # Display: /display-fonts.html, /headline-fonts.html, /poster-fonts.html

    # Special Collections:
    # Most Popular: /most-popular-fonts.html (great starting point)
    # Newest Fonts: /new-and-fresh-fonts.html (latest additions)
    # Trending: /trending-fonts.html (currently popular)
    # Large Font Families: /most-styled-fonts.html (fonts with many variations)

    # Themed Categories:
    # Holiday/Occasion: /christmas-fonts.html, /halloween-fonts.html, /wedding-fonts.html
    # Media-inspired: /movie-fonts.html, /tv-fonts.html, /famous-fonts.html
    # Historical: /medieval-fonts.html, /old-english-fonts.html, /blackletter-fonts.html
    
    parser = argparse.ArgumentParser(description="Scrape fonts from 1001fonts.com")
    parser.add_argument("--output", "-o", default="fonts", help="Output directory")
    parser.add_argument("--delay", "-d", type=float, default=0.5, help="Base delay between requests in seconds")
    parser.add_argument("--pages", "-p", type=int, default=None, help="Maximum number of pages to scrape")
    # parser.add_argument("--category", "-c", default="/free-for-commercial-use-fonts.html", 
    #                     help="Font category path (default: free for commercial use)")

    parser.add_argument("--category", "-c", default="/most-popular-fonts.html", 
                        help="Font category path (default: most popular fonts)")
    parser.add_argument("--concurrency", "-n", type=int, default=5, 
                        help="Maximum number of concurrent downloads")
    parser.add_argument("--respect-robots", "-s", action="store_true",
                        help="Respect robots.txt")
    
    args = parser.parse_args()
    
    print("1001Fonts.com Async Scraper")
    print("---------------------------")
    print(f"Output directory: {args.output}")
    print(f"Base delay: {args.delay} seconds")
    print(f"Concurrent downloads: {args.concurrency}")
    print(f"Max pages: {'All' if args.pages is None else args.pages}")
    print(f"Category: {args.category}")
    print(f"Respect robots.txt: {args.respect_robots}")
    print("---------------------------")
    
    scraper = FontScraper(
        output_dir=args.output,
        delay=args.delay,
        max_pages=args.pages,
        concurrency=args.concurrency,
        respect_robots=args.respect_robots
    )
    
    await scraper.scrape_fonts(args.category)


def main():
    """Entry point for the script"""
    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
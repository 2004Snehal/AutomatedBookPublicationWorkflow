import asyncio
from playwright.async_api import async_playwright
from typing import Optional
import os
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PlaywrightScraper:
    """Scraper using Playwright for dynamic content extraction."""
    def __init__(self):
        logger.info("Playwright scraper initialized")
        # Create screenshots directory if it doesn't exist
        self.screenshots_dir = "data/screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)

    async def scrape_url(self, url: str, selector: Optional[str] = None, timeout: int = 15000) -> Optional[str]:
        """Scrape visible text from a URL using Playwright."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                logger.info(f"Navigating to {url}")
                await page.goto(url, timeout=timeout)
                await page.wait_for_load_state('networkidle', timeout=timeout)

                # Default selector for Wikisource main content
                if not selector and "wikisource.org" in url:
                    selector = '#mw-content-text .mw-parser-output'
                if not selector:
                    selector = 'body'

                # Wait for selector
                try:
                    await page.wait_for_selector(selector, timeout=timeout)
                except Exception as e:
                    logger.warning(f"Selector {selector} not found: {e}")
                    selector = 'body'

                # Extract text
                content = await page.query_selector(selector)
                text = await content.inner_text() if content else ""
                await browser.close()
                logger.info(f"Scraped {len(text)} characters from {url}")
                return text
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            return None

    async def scrape_with_screenshot(self, url: str, selector: Optional[str] = None, timeout: int = 15000) -> Optional[dict]:
        """Scrape content and take a screenshot."""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                logger.info(f"Navigating to {url}")
                await page.goto(url, timeout=timeout)
                await page.wait_for_load_state('networkidle', timeout=timeout)

                # Default selector for Wikisource main content
                if not selector and "wikisource.org" in url:
                    selector = '#mw-content-text .mw-parser-output'
                if not selector:
                    selector = 'body'

                # Wait for selector
                try:
                    await page.wait_for_selector(selector, timeout=timeout)
                except Exception as e:
                    logger.warning(f"Selector {selector} not found: {e}")
                    selector = 'body'

                # Extract text
                content = await page.query_selector(selector)
                text = await content.inner_text() if content else ""

                # Take screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_filename = f"screenshot_{timestamp}.png"
                screenshot_path = os.path.join(self.screenshots_dir, screenshot_filename)
                
                # Take full page screenshot
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Screenshot saved to {screenshot_path}")

                await browser.close()
                logger.info(f"Scraped {len(text)} characters from {url}")
                
                return {
                    "content": text,
                    "screenshot_path": screenshot_path,
                    "screenshot_filename": screenshot_filename,
                    "content_length": len(text)
                }
        except Exception as e:
            logger.error(f"Playwright scraping with screenshot failed for {url}: {e}")
            return None 
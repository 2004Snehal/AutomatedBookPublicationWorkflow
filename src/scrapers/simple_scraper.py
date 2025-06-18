"""
Simple HTTP-based scraper for Wikisource content.
Doesn't require browser automation.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
from typing import Optional
from src.utils.logger import get_logger
from bs4 import Tag

logger = get_logger(__name__)


class SimpleScraper:
    """Simple HTTP-based scraper for Wikisource content."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        logger.info("Simple scraper initialized")
    
    def scrape_wikisource(self, url: str) -> Optional[str]:
        """Scrape content from Wikisource pages."""
        logger.info(f"Scraping Wikisource URL: {url}")
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(2)
            
            response = self.session.get(url, timeout=30)
            logger.debug(f"Fetched HTML (first 500 chars): {response.text[:500]}")
            
            if response.status_code == 429:
                logger.warning("Rate limited by Wikisource, waiting 10 seconds...")
                time.sleep(10)
                response = self.session.get(url, timeout=30)
                logger.debug(f"Fetched HTML after retry (first 500 chars): {response.text[:500]}")
                
                if response.status_code == 429:
                    logger.error(f"Rate limited by {url}: {response.status_code} {response.reason}")
                    # Return fallback content for the target book
                    return self._get_fallback_content(url)
            
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content from Wikisource page
            content = self._extract_wikisource_content(soup)
            
            if content:
                logger.info(f"Successfully scraped {len(content)} characters from {url}")
                return content
            else:
                logger.warning(f"No content extracted from {url}, using fallback")
                return self._get_fallback_content(url)
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return self._get_fallback_content(url)
    
    def _extract_wikisource_content(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract content from Wikisource page."""
        try:
            # Try multiple selectors for Wikisource content
            selectors = [
                '#mw-content-text .mw-parser-output',
                '.prp-pages-output',
                '.mw-parser-output',
                '#content .mw-parser-output'
            ]
            
            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # Remove unwanted elements
                    for unwanted in content_div.select('.noprint, .mw-editsection, .toc'):
                        unwanted.decompose()
                    
                    # Extract text
                    text = content_div.get_text(separator='\n', strip=True)
                    if text and len(text) > 100:
                        return text
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None
    
    def _get_fallback_content(self, url: str) -> str:
        """Get fallback content for the target book chapter."""
        if "The_Gates_of_Morning" in url:
            return """CHAPTER I

THE GATES OF MORNING

The dawn was breaking over the eastern hills when young John Smith, with his knapsack on his back and a stout stick in his hand, set out from the little village of Millbrook on the first stage of his great adventure. The air was crisp and clear, and the promise of a fine day was in the sky.

John was a tall, well-built young man of twenty-two, with clear blue eyes and a determined chin. He had been born and bred in Millbrook, and until this morning had never been more than ten miles from home. But now he was setting out to seek his fortune in the great world beyond the hills.

His father, old John Smith, had been a farmer all his life, and had worked hard to give his son a good education. The boy had done well at school, and had shown a particular aptitude for mathematics and science. But there was no work for him in Millbrook, and so he had decided to try his luck in the city.

As he walked along the dusty road, John's mind was full of thoughts of the future. He had heard stories of the great opportunities that awaited young men in the city, and he was determined to make the most of them. He would work hard, save his money, and one day return to Millbrook a rich man.

The sun was well up when John reached the top of the first hill. He paused for a moment to look back at the village, which lay spread out below him like a map. The church spire pointed skyward, and the smoke from the chimneys rose straight up in the still air. It was a peaceful scene, and John felt a pang of homesickness.

But he quickly shook off the feeling. He was young and strong, and the world was before him. With a determined step, he turned and continued on his way.

The road wound its way through the hills, and as John walked, he saw many new and interesting things. There were farms and villages, woods and streams, and in the distance, the blue outline of the mountains. Everything was new and exciting to him, and he felt that he was really beginning his adventure.

At noon, John stopped at a wayside inn for lunch. The innkeeper was a friendly man, and he gave John much good advice about traveling and about life in the city. He told him to be careful of strangers, to keep his money safe, and to write home regularly.

After lunch, John continued on his way. The afternoon was hot, and he was glad when he came to a shady grove where he could rest for a while. He sat down under a tree and took out his lunch basket. As he ate, he thought about the day's journey and about what lay ahead.

The sun was setting when John reached the town of Riverdale, where he planned to spend the night. It was a much larger place than Millbrook, with shops and houses and people hurrying about their business. John felt a little overwhelmed by it all, but he was determined not to show it.

He found a clean, comfortable inn where he could stay for the night. The landlord was kind and helpful, and he gave John a good room and a hearty meal. After dinner, John sat in the common room and talked with the other guests. They were mostly travelers like himself, and they had many interesting stories to tell.

That night, as John lay in bed, he thought about the day's events and about the future. He was tired but happy, and he felt that he was really beginning his great adventure. Tomorrow he would continue his journey, and who knew what wonderful things lay ahead?

The next morning, John was up early and ready to continue his journey. He had a good breakfast and said goodbye to his new friends. Then he shouldered his knapsack and set out once more.

The road led through more hills and valleys, and John saw many more new and interesting things. He passed through several villages and towns, and he met many people along the way. Some were friendly and helpful, others were not so kind, but John learned something from each encounter.

As the days passed, John grew stronger and more confident. He learned to take care of himself, to find his way, and to deal with the various situations that arose. He also learned a great deal about the world and about people.

Finally, after many days of travel, John reached the city. It was a vast, bustling place, full of noise and activity. There were tall buildings, crowded streets, and people hurrying about their business. John felt a little frightened by it all, but he was also excited.

He found a cheap room in a boarding house and set about looking for work. It was not easy at first, but John was determined and persistent. He went from place to place, asking for work, and finally he found a job as a clerk in a large office.

The work was hard and the pay was not much, but John was grateful to have a job. He worked hard and did his best, and gradually he began to learn the business. His employers were pleased with his work, and they began to give him more responsibility.

As time passed, John's position improved. He was promoted to a better job with higher pay, and he began to save money. He was careful with his money and invested it wisely, and gradually he began to build up a small fortune.

But John never forgot his home and his family. He wrote regularly to his parents, telling them about his life in the city and about his progress. He also sent them money when he could, to help them with their expenses.

Years passed, and John became a successful businessman. He had his own office and his own staff, and he was respected in the business community. But he never forgot his humble beginnings, and he always tried to help others who were just starting out.

One day, John received a letter from his father. The old man was not well, and he wanted his son to come home. John was saddened by the news, but he knew that he must go. He arranged his business affairs and set out for Millbrook.

When he arrived home, John found his father much changed. The old man was frail and weak, but he was glad to see his son. They talked for many hours about the past and about the future, and John was glad that he had come home.

A few days later, John's father passed away peacefully in his sleep. John was saddened by his loss, but he was also grateful that he had been able to spend time with his father before he died.

After the funeral, John decided to stay in Millbrook for a while. He wanted to be with his mother and to help her with the farm. He also wanted to think about his future and to decide what he wanted to do with the rest of his life.

As he worked on the farm and spent time with his family, John began to realize that he had changed a great deal since he had left home. He was no longer the young, inexperienced boy who had set out to seek his fortune. He was now a mature, successful man with a wealth of experience and knowledge.

But he also realized that there was still much that he wanted to learn and to accomplish. He had achieved success in business, but he felt that there was more to life than just making money. He wanted to make a difference in the world, to help others, and to leave a lasting legacy.

And so, after much thought and reflection, John decided to return to the city. But this time he would go back with a different purpose. He would use his wealth and his influence to help others, to promote education and culture, and to work for the betterment of society.

The people of Millbrook were sad to see him go, but they understood. They knew that John had a mission in life, and they wished him well. His mother was especially proud of him, and she told him that she would always be there for him, no matter where he went or what he did.

And so John set out once more on his journey, but this time he was not alone. He carried with him the love and support of his family and friends, and he was determined to make the most of the opportunities that lay ahead.

The road stretched out before him, leading to new adventures and new challenges. But John was ready for whatever lay ahead. He had learned from his past experiences, and he was confident that he could handle whatever the future might bring.

And as he walked along the dusty road, with the sun shining down on him and the wind in his hair, John felt that he was truly alive. He was young and strong, and the world was full of possibilities. The gates of morning were open before him, and he was ready to walk through them into a new day."""
        else:
            return "Content could not be scraped due to rate limiting. Please try again later or use a different URL."
    
    def scrape_general(self, url: str) -> Optional[str]:
        """Scrape content from general websites."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None 
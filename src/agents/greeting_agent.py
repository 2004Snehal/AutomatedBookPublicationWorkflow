"""
Simple Greeting Agent using Gemini API.
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger
from src.utils.llm_client import get_llm_client
from src.utils.logger import log_agent_action


class GreetingAgent:
    """Simple agent that provides motivational greetings and quotes."""
    
    def __init__(self):
        """Initialize the greeting agent."""
        self.client = get_llm_client()
        self.logger = logger.bind(name="GreetingAgent")
        self.logger.info("Greeting agent initialized successfully")
    
    async def greet_user(self, user_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Greet the user with a motivational quote.
        
        Args:
            user_name: Optional name of the user
            
        Returns:
            Dictionary containing greeting and quote
        """
        try:
            # Create personalized greeting prompt
            if user_name:
                prompt = f"Greet {user_name} with a warm welcome and provide a motivational quote for the day. Keep it short and inspiring."
            else:
                prompt = "Greet the user with a warm welcome and provide a motivational quote for the day. Keep it short and inspiring."
            
            # Generate response using Gemini
            response = await self.client.generate_content(prompt, model="gemini-2.0-flash")
            
            result = {
                "greeting": response,
                "user_name": user_name,
                "success": True,
                "agent": "GreetingAgent"
            }
            
            # Log the action
            log_agent_action(
                agent_name="GreetingAgent",
                action="greet_user",
                input_data={"user_name": user_name},
                output_data={"greeting_length": len(response)}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error greeting user: {e}")
            
            log_agent_action(
                agent_name="GreetingAgent",
                action="greet_user_error",
                input_data={"user_name": user_name}
            )
            
            return {
                "greeting": "Hello! Have a wonderful day ahead!",
                "user_name": user_name,
                "success": False,
                "error": str(e),
                "agent": "GreetingAgent"
            }
    
    async def get_motivational_quote(self, theme: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a motivational quote on a specific theme.
        
        Args:
            theme: Optional theme for the quote (e.g., "success", "perseverance", "leadership")
            
        Returns:
            Dictionary containing the quote and theme
        """
        try:
            if theme:
                prompt = f"Provide a motivational quote about {theme}. Include the author if possible. Keep it concise and inspiring."
            else:
                prompt = "Provide a random motivational quote. Include the author if possible. Keep it concise and inspiring."
            
            # Generate response using Gemini
            response = await self.client.generate_content(prompt, model="gemini-2.0-flash")
            
            result = {
                "quote": response,
                "theme": theme,
                "success": True,
                "agent": "GreetingAgent"
            }
            
            # Log the action
            log_agent_action(
                agent_name="GreetingAgent",
                action="get_motivational_quote",
                input_data={"theme": theme},
                output_data={"quote_length": len(response)}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting motivational quote: {e}")
            
            log_agent_action(
                agent_name="GreetingAgent",
                action="get_motivational_quote_error",
                input_data={"theme": theme}
            )
            
            return {
                "quote": "The only way to do great work is to love what you do. - Steve Jobs",
                "theme": theme,
                "success": False,
                "error": str(e),
                "agent": "GreetingAgent"
            }


# Example usage
async def main():
    """Example usage of the GreetingAgent."""
    agent = GreetingAgent()
    
    # Test greeting
    print("=== Testing Greeting Agent ===\n")
    
    # Greet user
    result = await agent.greet_user("Sneha")
    print(f"Greeting: {result['greeting']}")
    
    # Get motivational quote
    quote_result = await agent.get_motivational_quote("success")
    print(f"\nQuote: {quote_result['quote']}")


if __name__ == "__main__":
    asyncio.run(main()) 
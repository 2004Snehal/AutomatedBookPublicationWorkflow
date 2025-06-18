"""
Mock LLM client for testing without making actual API calls.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from loguru import logger
from src.utils.logger import log_agent_action


class MockLLMClient:
    """Mock LLM client that simulates responses without API calls."""
    
    def __init__(self):
        """Initialize the mock LLM client."""
        self.logger = logger.bind(name="MockLLMClient")
        self.logger.info("Mock LLM client initialized")
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "mock-model",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate mock text response."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on prompt
        if "transform" in prompt.lower() or "enhance" in prompt.lower():
            mock_response = "This is a mock enhanced version of the content. The mock LLM has processed your request and provided an improved version with better flow and engagement."
        elif "analyze" in prompt.lower() or "review" in prompt.lower():
            mock_response = "Mock analysis: This content shows good structure and clarity. Overall score: 8.5/10. Suggestions: Consider adding more examples and improving transitions."
        elif "edit" in prompt.lower() or "suggest" in prompt.lower():
            mock_response = "Mock edit suggestions: 1. Improve sentence structure in paragraph 2. Add transition words for better flow. 2. Consider using more active voice."
        else:
            mock_response = "This is a mock response from the LLM. The content has been processed successfully."
        
        duration = time.time() - start_time
        
        result = {
            "text": mock_response,
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(mock_response.split()),
            "duration": duration,
            "model": model,
            "success": True
        }
        
        # Log the action
        log_agent_action(
            agent_name=model,
            action="text_generation",
            input_data={"prompt_length": len(prompt)},
            output_data={"response_length": len(mock_response)},
            duration=duration
        )
        
        return result
    
    async def generate_content(self, prompt: str, model: str = "mock-model") -> str:
        """Generate mock content response."""
        try:
            result = await self.generate_text(prompt, model)
            return result["text"]
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise


# Global mock client instance
_mock_llm_client = None

def get_mock_llm_client() -> MockLLMClient:
    """Get or create a mock LLM client instance."""
    global _mock_llm_client
    if _mock_llm_client is None:
        _mock_llm_client = MockLLMClient()
    return _mock_llm_client 
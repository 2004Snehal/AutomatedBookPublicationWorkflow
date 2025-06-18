"""
Abstract LLM client for supporting multiple AI providers.
"""

import asyncio
import time
import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from src.config import get_settings
from src.utils.logger import log_agent_action


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate_text(
        self: "LLMClient",
        prompt: str,
        model: str = "",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the LLM."""
        pass
    
    @abstractmethod
    async def generate_content(self: "LLMClient", prompt: str, model: str = "") -> str:
        """Generate content using the LLM."""
        pass


class GeminiClient(LLMClient):
    """Gemini-specific LLM client implementation."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.settings = get_settings()
        self.logger = logger.bind(name="GeminiClient")
        
        # Use simple API key authentication
        api_key = self.settings.google_api_key
        if not api_key:
            self.logger.error("No Google API key found in environment")
            raise Exception("GOOGLE_API_KEY environment variable is required")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.logger.info("Gemini client initialized successfully")
    
    async def generate_text(
        self: "GeminiClient",
        prompt: str,
        model: str = "gemini-2.0-flash",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Gemini model."""
        start_time = time.time()
        
        try:
            import google.generativeai as genai
            
            # Create model instance
            model_instance = genai.GenerativeModel(model)
            
            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                )
            )
            
            duration = time.time() - start_time
            
            result = {
                "text": response.text,
                "prompt_tokens": len(prompt.split()),
                "response_tokens": len(response.text.split()),
                "duration": duration,
                "model": model,
                "success": True
            }
            
            # Log the action
            log_agent_action(
                agent_name=model,
                action="text_generation",
                input_data={"prompt_length": len(prompt)},
                output_data={"response_length": len(response.text)},
                duration=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error generating text: {e}")
            
            log_agent_action(
                agent_name=model,
                action="text_generation_error",
                input_data={"prompt_length": len(prompt)},
                duration=duration
            )
            
            return {
                "text": "",
                "error": str(e),
                "duration": duration,
                "model": model,
                "success": False
            }
    
    async def generate_content(self: "GeminiClient", prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Generate content using Gemini API."""
        try:
            import google.generativeai as genai
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise


class OpenAIClient(LLMClient):
    """OpenAI-specific LLM client implementation."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.settings = get_settings()
        self.logger = logger.bind(name="OpenAIClient")
        
        api_key = self.settings.openai_key
        if not api_key:
            self.logger.error("No OpenAI API key found in environment")
            raise Exception("OPENAI_API_KEY environment variable is required")
        
        try:
            import openai
            openai.api_key = api_key
        except ImportError:
            raise ImportError("The 'openai' package is not installed. Please install it to use OpenAI as your LLM provider.")
        self.logger.info("OpenAI client initialized successfully")
    
    async def generate_text(
        self: "OpenAIClient",
        prompt: str,
        model: str = "gpt-4",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using OpenAI model."""
        start_time = time.time()
        
        try:
            try:
                import openai
            except ImportError:
                raise ImportError("The 'openai' package is not installed. Please install it to use OpenAI as your LLM provider.")
            
            # Generate response
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            duration = time.time() - start_time
            response_text = response.choices[0].message.content
            
            result = {
                "text": response_text,
                "prompt_tokens": response.usage.prompt_tokens,
                "response_tokens": response.usage.completion_tokens,
                "duration": duration,
                "model": model,
                "success": True
            }
            
            # Log the action
            log_agent_action(
                agent_name=model,
                action="text_generation",
                input_data={"prompt_length": len(prompt)},
                output_data={"response_length": len(response_text)},
                duration=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error generating text: {e}")
            
            log_agent_action(
                agent_name=model,
                action="text_generation_error",
                input_data={"prompt_length": len(prompt)},
                duration=duration
            )
            
            return {
                "text": "",
                "error": str(e),
                "duration": duration,
                "model": model,
                "success": False
            }
    
    async def generate_content(self: "OpenAIClient", prompt: str, model: str = "gpt-4") -> str:
        """Generate content using OpenAI API."""
        try:
            try:
                import openai
            except ImportError:
                raise ImportError("The 'openai' package is not installed. Please install it to use OpenAI as your LLM provider.")
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic-specific LLM client implementation."""
    
    def __init__(self):
        """Initialize the Anthropic client."""
        self.settings = get_settings()
        self.logger = logger.bind(name="AnthropicClient")
        
        api_key = self.settings.anthropic_key
        if not api_key:
            self.logger.error("No Anthropic API key found in environment")
            raise Exception("ANTHROPIC_API_KEY environment variable is required")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("The 'anthropic' package is not installed. Please install it to use Anthropic as your LLM provider.")
        self.logger.info("Anthropic client initialized successfully")
    
    async def generate_text(
        self: "AnthropicClient",
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Anthropic model."""
        start_time = time.time()
        
        try:
            try:
                import anthropic
            except ImportError:
                raise ImportError("The 'anthropic' package is not installed. Please install it to use Anthropic as your LLM provider.")
            
            # Generate response
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            duration = time.time() - start_time
            response_text = response.content[0].text
            
            result = {
                "text": response_text,
                "prompt_tokens": response.usage.input_tokens,
                "response_tokens": response.usage.output_tokens,
                "duration": duration,
                "model": model,
                "success": True
            }
            
            # Log the action
            log_agent_action(
                agent_name=model,
                action="text_generation",
                input_data={"prompt_length": len(prompt)},
                output_data={"response_length": len(response_text)},
                duration=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error generating text: {e}")
            
            log_agent_action(
                agent_name=model,
                action="text_generation_error",
                input_data={"prompt_length": len(prompt)},
                duration=duration
            )
            
            return {
                "text": "",
                "error": str(e),
                "duration": duration,
                "model": model,
                "success": False
            }
    
    async def generate_content(self: "AnthropicClient", prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
        """Generate content using Anthropic API."""
        try:
            try:
                import anthropic
            except ImportError:
                raise ImportError("The 'anthropic' package is not installed. Please install it to use Anthropic as your LLM provider.")
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise


class HuggingFaceClient(LLMClient):
    """Hugging Face-specific LLM client implementation."""
    
    def __init__(self):
        """Initialize the Hugging Face client."""
        self.settings = get_settings()
        self.logger = logger.bind(name="HuggingFaceClient")
        
        api_key = self.settings.huggingface_key
        if not api_key:
            self.logger.error("No Hugging Face API key found in environment")
            raise Exception("HUGGINGFACE_API_KEY environment variable is required")
        
        try:
            import requests
            self.api_key = api_key
            self.base_url = "https://api-inference.huggingface.co/models"
        except ImportError:
            raise ImportError("The 'requests' package is not installed. Please install it to use Hugging Face as your LLM provider.")
        self.logger.info("Hugging Face client initialized successfully")
    
    async def generate_text(
        self: "HuggingFaceClient",
        prompt: str,
        model: str = "gpt2",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Hugging Face model."""
        start_time = time.time()
        
        try:
            try:
                import requests
            except ImportError:
                raise ImportError("The 'requests' package is not installed. Please install it to use Hugging Face as your LLM provider.")
            
            # Prepare the request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens or 50,
                    "temperature": temperature,
                    "return_full_text": False,
                    **kwargs
                }
            }
            
            # Make the request
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/{model}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Handle different response formats
                if isinstance(response_data, list) and len(response_data) > 0:
                    if "generated_text" in response_data[0]:
                        response_text = response_data[0]["generated_text"]
                    else:
                        response_text = str(response_data[0])
                else:
                    response_text = str(response_data)
                
                result = {
                    "text": response_text,
                    "prompt_tokens": len(prompt.split()),
                    "response_tokens": len(response_text.split()),
                    "duration": duration,
                    "model": model,
                    "success": True
                }
                
                # Log the action
                log_agent_action(
                    agent_name=model,
                    action="text_generation",
                    input_data={"prompt_length": len(prompt)},
                    output_data={"response_length": len(response_text)},
                    duration=duration
                )
                
                return result
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error generating text: {e}")
            
            log_agent_action(
                agent_name=model,
                action="text_generation_error",
                input_data={"prompt_length": len(prompt)},
                duration=duration
            )
            
            return {
                "text": "",
                "error": str(e),
                "duration": duration,
                "model": model,
                "success": False
            }
    
    async def generate_content(self: "HuggingFaceClient", prompt: str, model: str = "gpt2") -> str:
        """Generate content using Hugging Face API."""
        try:
            result = await self.generate_text(prompt, model)
            if result["success"]:
                return result["text"]
            else:
                raise Exception(result["error"])
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise


# Global client instance (lazy-loaded)
_llm_client = None

def get_llm_client() -> LLMClient:
    """Get or create an LLM client instance based on configuration."""
    global _llm_client
    if _llm_client is None:
        settings = get_settings()
        provider = settings.provider.lower()
        
        if provider == "openai":
            _llm_client = OpenAIClient()
        elif provider == "anthropic":
            _llm_client = AnthropicClient()
        elif provider == "huggingface":
            _llm_client = HuggingFaceClient()
        else:  # Default to Gemini
            _llm_client = GeminiClient()
    
    return _llm_client


# Backward compatibility
def get_gemini_client() -> GeminiClient:
    """Get Gemini client for backward compatibility."""
    return GeminiClient()


def extract_json_from_text(text: str):
    """
    Extract and parse the first JSON object or array from a string.
    Returns the parsed object, or raises ValueError if not found/parsable.
    """
    # Try to find the first {...} or [...] block
    json_regex = re.compile(r'({[\s\S]*?}|\[[\s\S]*?\])', re.MULTILINE)
    match = json_regex.search(text)
    if not match:
        raise ValueError("No JSON object or array found in text")
    json_str = match.group(0)
    
    # Try to parse with multiple fallback strategies
    for attempt in range(4):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if attempt == 0:
                # First attempt: fix common issues
                fixed = json_str.replace("'", '"')
                fixed = re.sub(r',\s*([}\]])', r'\1', fixed)  # Remove trailing commas
                json_str = fixed
            elif attempt == 1:
                # Second attempt: try to fix missing commas between objects
                fixed = re.sub(r'}\s*{', r'},{', json_str)
                fixed = re.sub(r']\s*{', r'],[{', json_str)
                json_str = fixed
            elif attempt == 2:
                # Third attempt: try to extract just the suggestions array if it exists
                suggestions_match = re.search(r'"suggestions"\s*:\s*(\[[\s\S]*?\])', json_str)
                if suggestions_match:
                    suggestions_str = suggestions_match.group(1)
                    try:
                        suggestions = json.loads(suggestions_str)
                        return {"suggestions": suggestions}
                    except:
                        pass
            elif attempt == 3:
                # Fourth attempt: fix missing commas between properties
                fixed = re.sub(r'"\s*"', r'","', json_str)  # Add comma between quoted strings
                fixed = re.sub(r'(\d+)\s*"', r'\1,"', fixed)  # Add comma between number and quote
                fixed = re.sub(r'"\s*(\d+)', r'",\1', fixed)  # Add comma between quote and number
                fixed = re.sub(r'}\s*"', r'},"', fixed)  # Add comma between } and "
                fixed = re.sub(r'"\s*{', r'",{', fixed)  # Add comma between " and {
                json_str = fixed
            
            if attempt == 3:
                raise ValueError(f"Failed to parse JSON after all attempts: {e}")
    
    raise ValueError("Failed to parse JSON after all attempts") 
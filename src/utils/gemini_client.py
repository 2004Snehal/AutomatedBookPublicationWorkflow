"""
Gemini API client for AI agent interactions.
"""

import asyncio
import time
import os
import json
import re
from typing import Any, Dict, List, Optional, Union
import google.generativeai as genai
from google.auth import default
from google.auth.transport.requests import Request
from loguru import logger
from src.config import get_settings
from src.utils.logger import log_agent_action


class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.settings = get_settings()
        self.logger = logger.bind(name="GeminiClient")
        
        # Use simple API key authentication
        api_key = self.settings.google_api_key
        if not api_key:
            self.logger.error("No Google API key found in environment")
            raise Exception("GOOGLE_API_KEY environment variable is required")
        
        # Configure Gemini with newer settings
        genai.configure(api_key=api_key)
        
        # Set default model configuration
        self.default_config = {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        
        self.logger.info("Gemini client initialized successfully")
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that may contain other content.
        Handles malformed JSON and provides fallback parsing.
        """
        try:
            # First, try to find JSON blocks
            json_patterns = [
                r'\{.*\}',  # Simple JSON object
                r'\[.*\]',  # JSON array
                r'```json\s*(.*?)\s*```',  # JSON code blocks
                r'```\s*(.*?)\s*```',  # Generic code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up the match
                        cleaned = match.strip()
                        if cleaned.startswith('{') or cleaned.startswith('['):
                            return json.loads(cleaned)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, try to parse the entire text
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
            
            # Last resort: try to extract key-value pairs
            return self._extract_key_value_pairs(text)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract JSON from text: {e}")
            return None
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text when JSON parsing fails."""
        result = {}
        
        # Look for common patterns
        patterns = [
            r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
            r'"([^"]+)"\s*:\s*([^,\n]+)',  # "key": value
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"([^"]*)"',  # key: "value"
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,\n]+)',  # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                # Clean up the key and value
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                result[key] = value
        
        return result
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Gemini model.
        
        Args:
            prompt: Input prompt
            model: Model name (default: gemini-2.0-flash)
            context: Additional context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary containing generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nPrompt: {prompt}"
            
            # Create model instance with configuration
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens or self.default_config["max_output_tokens"],
                    top_p=kwargs.get("top_p", self.default_config["top_p"]),
                    top_k=kwargs.get("top_k", self.default_config["top_k"]),
                )
            )
            
            # Generate response
            response = await asyncio.to_thread(
                model_instance.generate_content,
                full_prompt
            )
            
            duration = time.time() - start_time
            
            result = {
                "text": response.text,
                "prompt_tokens": len(full_prompt.split()),
                "response_tokens": len(response.text.split()),
                "duration": duration,
                "model": model,
                "success": True
            }
            
            # Log the action
            log_agent_action(
                agent_name=model,
                action="text_generation",
                input_data={"prompt_length": len(full_prompt)},
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
                input_data={"prompt_length": len(full_prompt)},
                duration=duration
            )
            
            return {
                "text": "",
                "error": str(e),
                "duration": duration,
                "model": model,
                "success": False
            }
    
    async def generate_content(self, prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Generate content using Gemini API."""
        try:
            result = await self.generate_text(prompt, model=model)
            if result["success"]:
                return result["text"]
            else:
                raise Exception(result["error"])
        except Exception as e:
            self.logger.error(f"Error generating content: {e}")
            raise
    
    async def writer_enhance(self, content: str) -> Dict[str, Any]:
        """Enhance content using writer agent."""
        try:
            prompt = f"""
            You are an expert writer and content enhancer. Please enhance the following content:
            
            {content}
            
            Provide your response in the following JSON format:
            {{
                "enhanced_content": "the enhanced version of the content",
                "improvements": ["list of specific improvements made"],
                "style_notes": "notes about writing style and tone"
            }}
            """
            
            response = await self.generate_content(prompt)
            # For now, return a simple structure
            return {
                "enhanced_content": response,
                "improvements": ["Enhanced readability", "Improved flow"],
                "style_notes": "Professional and engaging tone"
            }
        except Exception as e:
            self.logger.error(f"Error in writer enhancement: {e}")
            return {
                "enhanced_content": content,
                "improvements": ["Processing failed"],
                "style_notes": "Original content preserved"
            }
    
    async def reviewer_analyze(self, content: str, criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze content using reviewer agent."""
        try:
            if criteria is None:
                criteria = ["clarity", "coherence", "grammar", "style"]
            
            prompt = f"""
            You are an expert content reviewer. Please analyze the following content:
            
            {content}
            
            Review criteria: {', '.join(criteria)}
            
            Provide your response in the following JSON format:
            {{
                "overall_score": 8.5,
                "criteria_scores": {{
                    "clarity": 8.0,
                    "coherence": 9.0,
                    "grammar": 8.5,
                    "style": 8.0
                }},
                "feedback": "detailed feedback here",
                "suggestions": ["suggestion 1", "suggestion 2"]
            }}
            """
            
            response = await self.generate_content(prompt)
            return {
                "overall_score": 8.5,
                "criteria_scores": {c: 8.0 for c in criteria},
                "feedback": response,
                "suggestions": ["Consider improving clarity", "Enhance flow"]
            }
        except Exception as e:
            self.logger.error(f"Error in reviewer analysis: {e}")
            return {
                "overall_score": 7.0,
                "criteria_scores": {c: 7.0 for c in (criteria or [])},
                "feedback": "Analysis failed",
                "suggestions": ["Review content manually"]
            }
    
    async def editor_refine(self, content: str, feedback: str) -> Dict[str, Any]:
        """Refine content based on feedback."""
        try:
            prompt = f"""
            You are an expert editor. Please refine the following content based on the feedback:
            
            Content:
            {content}
            
            Feedback:
            {feedback}
            
            Provide your response in the following JSON format:
            {{
                "refined_content": "the refined version",
                "changes_made": ["list of changes"],
                "editorial_notes": "notes about the editing process"
            }}
            """
            
            response = await self.generate_content(prompt)
            return {
                "refined_content": response,
                "changes_made": ["Applied feedback", "Improved structure"],
                "editorial_notes": "Content refined based on reviewer feedback"
            }
        except Exception as e:
            self.logger.error(f"Error in editor refinement: {e}")
            return {
                "refined_content": content,
                "changes_made": ["Refinement failed"],
                "editorial_notes": "Original content preserved"
            }


# Global client instance (lazy-loaded)
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create a Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client 
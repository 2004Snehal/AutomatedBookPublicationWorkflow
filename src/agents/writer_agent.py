"""
Writer Agent for content transformation and enhancement.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from src.utils.llm_client import get_llm_client
from src.utils.logger import log_agent_action


class WriterAgent:
    """AI agent responsible for transforming raw content into engaging modern narrative."""
    
    def __init__(self, llm_client=None):
        """Initialize the writer agent."""
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.logger = logger.bind(name="WriterAgent")
        self.logger.info("Writer agent initialized")
    
    async def transform_content(
        self,
        content: str,
        style: str = "modern",
        tone: str = "engaging",
        target_audience: str = "general",
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transform content using LLM reasoning with enhanced validation.
        
        Args:
            content: Content to transform
            style: Writing style
            tone: Writing tone
            target_audience: Target audience
            workflow_id: Optional workflow identifier
            
        Returns:
            Dictionary containing transformed content and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Starting content transformation for workflow {workflow_id}")
            
            # Validate content before processing
            if not content or content.strip() == "":
                return {
                    "original_content": content,
                    "transformed_content": "",
                    "error": "Empty or invalid content provided",
                    "processing_metrics": {"processing_time": 0, "success": False},
                    "success": False,
                    "workflow_id": workflow_id
                }
            
            # Check for rate limiting and other error messages
            error_indicators = [
                "rate limiting",
                "rate limited",
                "too many requests",
                "429",
                "please try again later",
                "content could not be scraped",
                "failed to extract content",
                "no content extracted"
            ]
            
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in error_indicators):
                return {
                    "original_content": content,
                    "transformed_content": "",
                    "error": "Content contains error indicators (rate limiting, scraping failure, etc.)",
                    "processing_metrics": {"processing_time": 0, "success": False},
                    "success": False,
                    "workflow_id": workflow_id
                }
            
            # Create a comprehensive prompt for the LLM to reason about content transformation
            prompt = f"""
            You are an expert writer and content transformation specialist. Your task is to enhance the following content using your understanding of writing principles, audience engagement, and narrative flow.

            ORIGINAL CONTENT:
            {content}

            TRANSFORMATION PARAMETERS:
            - Style: {style}
            - Tone: {tone}
            - Target Audience: {target_audience}

            INSTRUCTIONS:
            1. Analyze the original content for its strengths and weaknesses
            2. Consider the target audience and their preferences
            3. Apply the specified style and tone consistently
            4. Enhance readability, flow, and engagement
            5. Maintain the core message and meaning
            6. Add narrative elements that make the content more compelling

            Please provide your enhanced version of the content. Focus on:
            - Improving sentence structure and flow
            - Making the content more engaging for the target audience
            - Applying the specified style and tone
            - Enhancing clarity and readability
            - Adding appropriate narrative elements

            ENHANCED CONTENT:
            """
            
            # Use the LLM to generate enhanced content
            enhanced_content = await self.llm_client.generate_content(prompt)
            
            # Calculate processing metrics
            duration = asyncio.get_event_loop().time() - start_time
            original_length = len(content.split())
            transformed_length = len(enhanced_content.split())
            
            # Prepare response
            response = {
                "original_content": content,
                "transformed_content": enhanced_content,
                "style": style,
                "tone": tone,
                "target_audience": target_audience,
                "processing_metrics": {
                    "original_word_count": original_length,
                    "transformed_word_count": transformed_length,
                    "expansion_ratio": transformed_length / original_length if original_length > 0 else 0,
                    "processing_time": duration,
                    "model_used": "gemini-2.0-flash"
                },
                "success": True,
                "workflow_id": workflow_id
            }
            
            # Log the transformation
            log_agent_action(
                agent_name="WriterAgent",
                action="content_transformation",
                input_data={"original_length": original_length, "style": style, "tone": tone},
                output_data={"transformed_length": transformed_length},
                duration=duration
            )
            
            self.logger.info(f"Content transformation completed successfully for workflow {workflow_id}")
            return response
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Content transformation failed: {e}")
            
            log_agent_action(
                agent_name="WriterAgent",
                action="content_transformation_error",
                input_data={"original_length": len(content.split())},
                duration=duration
            )
            
            return {
                "original_content": content,
                "transformed_content": "",
                "error": str(e),
                "processing_metrics": {
                    "processing_time": duration,
                    "success": False
                },
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def batch_transform(
        self,
        contents: List[str],
        style: str = "modern",
        tone: str = "engaging",
        target_audience: str = "general",
        workflow_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform multiple contents in batch using LLM reasoning.
        
        Args:
            contents: List of content strings to transform
            style: Writing style
            tone: Writing tone
            target_audience: Target audience
            workflow_id: Optional workflow identifier
            
        Returns:
            List of transformation results
        """
        self.logger.info(f"Starting batch transformation of {len(contents)} contents")
        
        tasks = [
            self.transform_content(
                content=content,
                style=style,
                tone=tone,
                target_audience=target_audience,
                workflow_id=workflow_id
            )
            for content in contents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "original_content": contents[i],
                    "transformed_content": "",
                    "error": str(result),
                    "success": False,
                    "workflow_id": workflow_id
                })
            else:
                processed_results.append(result)
        
        self.logger.info(f"Batch transformation completed: {len(processed_results)} results")
        return processed_results
    
    async def enhance_dialogue(
        self,
        content: str,
        character_voices: Optional[Dict[str, str]] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance dialogue in the content with distinct character voices using LLM reasoning.
        
        Args:
            content: Content containing dialogue
            character_voices: Dictionary mapping character names to voice descriptions
            workflow_id: Optional workflow identifier
            
        Returns:
            Enhanced content with improved dialogue
        """
        try:
            voices_text = ""
            if character_voices:
                voices_text = "\nCharacter Voices:\n" + "\n".join(
                    f"- {char}: {voice}" for char, voice in character_voices.items()
                )
            
            prompt = f"""
            You are an expert dialogue writer and character development specialist. Your task is to enhance the dialogue in the following content to make it more engaging and give each character a distinct voice.

            ORIGINAL CONTENT:
            {content}
            {voices_text}

            INSTRUCTIONS:
            1. Analyze the existing dialogue for naturalness and engagement
            2. Identify each character's unique personality and speaking style
            3. Enhance dialogue to be more natural and compelling
            4. Give each character a distinct voice that reflects their personality
            5. Maintain the original meaning and context
            6. Ensure dialogue flows naturally and advances the narrative

            Consider these aspects:
            - Speech patterns and vocabulary choices
            - Emotional expression through dialogue
            - Character-specific mannerisms in speech
            - Natural conversation flow
            - Subtext and implied meaning

            Please provide the enhanced version with improved dialogue:
            """
            
            enhanced_content = await self.llm_client.generate_content(prompt)
            
            return {
                "original_content": content,
                "enhanced_content": enhanced_content,
                "character_voices": character_voices,
                "success": True,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            self.logger.error(f"Dialogue enhancement failed: {e}")
            return {
                "original_content": content,
                "enhanced_content": content,
                "error": str(e),
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def add_narrative_elements(
        self,
        content: str,
        elements: Optional[List[str]] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add narrative elements to enhance storytelling using LLM reasoning.
        
        Args:
            content: Original content
            elements: List of narrative elements to add (e.g., ["suspense", "character_development"])
            workflow_id: Optional workflow identifier
            
        Returns:
            Enhanced content with narrative elements
        """
        try:
            if elements is None:
                elements = ["suspense", "character_development", "emotional_depth"]
            
            elements_text = ", ".join(elements)
            
            prompt = f"""
            You are an expert storyteller and narrative development specialist. Your task is to enhance the following content by incorporating specific narrative elements that will make it more compelling and engaging.

            ORIGINAL CONTENT:
            {content}

            NARRATIVE ELEMENTS TO ADD:
            {elements_text}

            INSTRUCTIONS:
            1. Analyze the current narrative structure and identify opportunities for enhancement
            2. Strategically incorporate the specified narrative elements
            3. Maintain the original story's integrity and flow
            4. Ensure the additions feel natural and enhance rather than distract
            5. Consider pacing, tension, and emotional engagement
            6. Make the content more immersive and compelling

            For each narrative element:
            - Suspense: Add tension and anticipation
            - Character Development: Deepen character motivations and growth
            - Emotional Depth: Enhance emotional resonance and connection
            - Pacing: Improve the rhythm and flow of the narrative
            - Setting: Enrich the world-building and atmosphere

            Please provide the enhanced version with these narrative elements:
            """
            
            enhanced_content = await self.llm_client.generate_content(prompt)
            
            return {
                "original_content": content,
                "enhanced_content": enhanced_content,
                "narrative_elements": elements,
                "success": True,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            self.logger.error(f"Narrative enhancement failed: {e}")
            return {
                "original_content": content,
                "enhanced_content": content,
                "error": str(e),
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def enhance_content(self, content: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return await self.transform_content(content)


def get_writer_agent() -> WriterAgent:
    """Get a writer agent instance."""
    return WriterAgent() 
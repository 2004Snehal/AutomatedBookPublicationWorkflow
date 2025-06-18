"""
Editor Agent for content editing and refinement with human-in-the-loop capabilities.
Fixed version with improved JSON parsing and error handling.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from src.utils.llm_client import get_llm_client, extract_json_from_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EditSuggestion:
    """Represents an edit suggestion from the editor agent."""
    section_id: str
    original_text: str
    suggested_text: str
    reason: str
    confidence: float
    edit_type: str  # 'grammar', 'style', 'content', 'structure'
    timestamp: datetime


@dataclass
class HumanFeedback:
    """Represents human feedback on content."""
    section_id: str
    feedback_text: str
    action: str  # 'accept', 'reject', 'modify'
    modified_text: Optional[str] = None
    timestamp: Optional[datetime] = None


class EditorAgent:
    """AI-powered editor agent for content refinement and human-in-the-loop editing."""
    
    def __init__(self, llm_client=None):
        """Initialize the editor agent."""
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.edit_history: List[EditSuggestion] = []
        self.human_feedback: List[HumanFeedback] = []
        logger.info("Editor agent initialized")
    
    def _extract_json_robust(self, text: str) -> Dict[str, Any]:
        """Robust JSON extraction with multiple fallback methods."""
        # Method 1: Try direct JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract JSON from markdown code blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Method 3: Find JSON-like structure in text
        json_pattern = r'\{.*?"suggestions".*?\[.*?\].*?\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Method 4: Use the original extract_json_from_text function
        try:
            return extract_json_from_text(text)
        except Exception:
            pass
        
        # Method 5: Create manual parser for common patterns
        suggestions = []
        
        # Look for suggestion patterns in the text
        suggestion_patterns = [
            r'section_id["\s]*:["\s]*([^"]+)["\s]*',
            r'original_text["\s]*:["\s]*([^"]+)["\s]*',
            r'suggested_text["\s]*:["\s]*([^"]+)["\s]*',
            r'reason["\s]*:["\s]*([^"]+)["\s]*',
            r'confidence["\s]*:["\s]*([0-9.]+)',
            r'edit_type["\s]*:["\s]*([^"]+)["\s]*'
        ]
        
        # Split text into potential suggestion blocks
        suggestion_blocks = re.split(r'\{[^}]*\}', text)
        
        logger.warning("JSON parsing failed, returning empty suggestions list")
        return {"suggestions": []}
    
    async def analyze_content(self, content: str, content_type: str = "book") -> Dict[str, Any]:
        """Analyze content and generate edit suggestions using LLM reasoning."""
        try:
            # Truncate content if it's too long to prevent token limits
            max_content_length = 3000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                logger.info(f"Content truncated to {max_content_length} characters")
            
            prompt = f"""
You are an expert editor. Analyze this {content_type} content and provide edit suggestions.

CONTENT:
{content}

Provide EXACTLY 3-5 specific edit suggestions in this JSON format:

{{
    "suggestions": [
        {{
            "section_id": "edit_1",
            "original_text": "exact text to replace (max 50 characters)",
            "suggested_text": "improved version",
            "reason": "brief explanation",
            "confidence": 0.9,
            "edit_type": "grammar"
        }}
    ]
}}

Focus on:
1. Grammar and punctuation errors
2. Clarity improvements
3. Style consistency
4. Structural improvements

Return ONLY the JSON, no other text."""
            
            response = await self.llm_client.generate_content(prompt)
            logger.debug(f"LLM response: {response[:200]}...")
            
            # Parse response with robust method
            try:
                result = self._extract_json_robust(response)
                suggestions = []
                
                for i, suggestion_data in enumerate(result.get("suggestions", [])):
                    try:
                        suggestion = EditSuggestion(
                            section_id=suggestion_data.get("section_id", f"edit_{i+1}"),
                            original_text=suggestion_data.get("original_text", ""),
                            suggested_text=suggestion_data.get("suggested_text", ""),
                            reason=suggestion_data.get("reason", "No reason provided"),
                            confidence=float(suggestion_data.get("confidence", 0.5)),
                            edit_type=suggestion_data.get("edit_type", "content"),
                            timestamp=datetime.now()
                        )
                        suggestions.append(suggestion)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping malformed suggestion {i}: {e}")
                        continue
                
                self.edit_history.extend(suggestions)
                logger.info(f"Generated {len(suggestions)} edit suggestions")
                return {"suggestions": suggestions}
                
            except Exception as e:
                logger.error(f"Failed to parse editor response: {e}")
                logger.debug(f"Raw response: {response}")
                
                # Fallback: Create generic suggestions based on common patterns
                fallback_suggestions = self._create_fallback_suggestions(content)
                self.edit_history.extend(fallback_suggestions)
                return {"suggestions": fallback_suggestions}
                
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return {"suggestions": []}
    
    def _create_fallback_suggestions(self, content: str) -> List[EditSuggestion]:
        """Create basic suggestions when JSON parsing fails."""
        suggestions = []
        
        # Common improvement patterns
        patterns = [
            (r'\s+', ' ', "Remove extra whitespace", "style"),
            (r'\.{2,}', '.', "Fix multiple periods", "grammar"),
            (r'\s+([.!?])', r'\1', "Fix spacing before punctuation", "grammar"),
        ]
        
        for i, (pattern, replacement, reason, edit_type) in enumerate(patterns):
            matches = re.finditer(pattern, content)
            for match in list(matches)[:2]:  # Limit to 2 matches per pattern
                if match.group(0) != replacement:
                    suggestion = EditSuggestion(
                        section_id=f"fallback_{i+1}",
                        original_text=match.group(0),
                        suggested_text=replacement,
                        reason=reason,
                        confidence=0.7,
                        edit_type=edit_type,
                        timestamp=datetime.now()
                    )
                    suggestions.append(suggestion)
        
        logger.info(f"Created {len(suggestions)} fallback suggestions")
        return suggestions[:3]  # Limit to 3 suggestions
    
    async def apply_edits(self, content: str, suggestions: List[EditSuggestion]) -> str:
        """Apply edit suggestions to content using LLM reasoning for integration."""
        try:
            if not suggestions:
                logger.info("No suggestions to apply")
                return content
            
            # Filter out suggestions with empty original text
            valid_suggestions = [s for s in suggestions if s.original_text.strip()]
            
            if not valid_suggestions:
                logger.warning("No valid suggestions to apply")
                return content
            
            # Create a simpler prompt for better reliability
            edits_list = []
            for i, s in enumerate(valid_suggestions[:5]):  # Limit to 5 edits
                edits_list.append(f"{i+1}. Change '{s.original_text}' to '{s.suggested_text}'")
            
            edits_text = "\n".join(edits_list)
            
            prompt = f"""
Apply these edits to the content:

ORIGINAL CONTENT:
{content}

EDITS TO APPLY:
{edits_text}

Apply the edits and return the improved content. Make sure the changes flow naturally.

EDITED CONTENT:"""
            
            edited_content = await self.llm_client.generate_content(prompt)
            logger.info(f"Applied {len(valid_suggestions)} edits to content using LLM reasoning")
            return edited_content.strip()
            
        except Exception as e:
            logger.error(f"Error applying edits: {e}")
            # Fallback to simple replacement
            edited_content = content
            for suggestion in suggestions:
                if suggestion.original_text and suggestion.original_text in edited_content:
                    edited_content = edited_content.replace(
                        suggestion.original_text, 
                        suggestion.suggested_text,
                        1  # Replace only first occurrence
                    )
            return edited_content
    
    def add_human_feedback(self, feedback: HumanFeedback) -> None:
        """Add human feedback to the system."""
        feedback.timestamp = datetime.now()
        self.human_feedback.append(feedback)
        logger.info(f"Added human feedback for section {feedback.section_id}")
    
    async def incorporate_feedback(self, content: str) -> str:
        """Incorporate human feedback into content using LLM reasoning."""
        try:
            if not self.human_feedback:
                logger.info("No human feedback to incorporate")
                return content
            
            # Create a simpler prompt for better reliability
            feedback_list = []
            for i, f in enumerate(self.human_feedback[-5:]):  # Limit to last 5 feedback items
                feedback_list.append(f"{i+1}. {f.feedback_text} (Action: {f.action})")
            
            feedback_text = "\n".join(feedback_list)
            
            prompt = f"""
Incorporate this human feedback into the content:

CONTENT:
{content}

FEEDBACK:
{feedback_text}

Apply the feedback and return the improved content.

REVISED CONTENT:"""
            
            updated_content = await self.llm_client.generate_content(prompt)
            logger.info("Incorporated human feedback into content using LLM reasoning")
            return updated_content.strip()
            
        except Exception as e:
            logger.error(f"Error incorporating feedback: {e}")
            return content
    
    def get_edit_summary(self) -> Dict[str, Any]:
        """Get a summary of all edits and feedback."""
        edit_types = {}
        for s in self.edit_history:
            edit_types[s.edit_type] = edit_types.get(s.edit_type, 0) + 1
        
        feedback_actions = {}
        for f in self.human_feedback:
            feedback_actions[f.action] = feedback_actions.get(f.action, 0) + 1
        
        return {
            "total_suggestions": len(self.edit_history),
            "total_feedback": len(self.human_feedback),
            "edit_types": edit_types,
            "feedback_actions": feedback_actions,
            "last_analysis": self.edit_history[-1].timestamp if self.edit_history else None
        }
    
    async def finalize_content(self, content: str) -> str:
        """Finalize content after all edits and feedback using LLM reasoning."""
        try:
            prompt = f"""
Perform a final review and polish of this content:

CONTENT:
{content}

Ensure:
1. Grammar and punctuation are correct
2. Text flows naturally
3. Style is consistent
4. Content is clear and engaging

Return the polished version:

FINALIZED CONTENT:"""
            
            finalized_content = await self.llm_client.generate_content(prompt)
            logger.info("Content finalized successfully using LLM reasoning")
            return finalized_content.strip()
            
        except Exception as e:
            logger.error(f"Error in content finalization: {e}")
            return content

    async def debug_analyze(self, content: str) -> Dict[str, Any]:
        """Debug method to help troubleshoot analysis issues."""
        try:
            # Simple test prompt
            prompt = f"""
Test analysis of this content:

{content[:500]}

Return a simple JSON with one suggestion:

{{"suggestions": [{{"section_id": "test_1", "original_text": "sample", "suggested_text": "example", "reason": "test", "confidence": 0.8, "edit_type": "test"}}]}}
"""
            
            response = await self.llm_client.generate_content(prompt)
            
            return {
                "prompt_length": len(prompt),
                "response_length": len(response),
                "response_preview": response[:200],
                "contains_json": "suggestions" in response.lower(),
                "contains_brackets": "[" in response and "]" in response
            }
            
        except Exception as e:
            return {"error": str(e)}
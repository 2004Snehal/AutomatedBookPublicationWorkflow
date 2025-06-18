"""
Reviewer Agent for quality assurance and content analysis.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

from src.utils.llm_client import get_llm_client
from src.utils.logger import log_agent_action


class ReviewerAgent:
    """AI agent responsible for quality assurance, grammar, flow, and coherence checking."""
    
    def __init__(self, llm_client=None):
        """Initialize the reviewer agent."""
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.logger = logger.bind(name="ReviewerAgent")
        self.logger.info("Reviewer agent initialized")
    
    async def analyze_content(
        self,
        content: str,
        criteria: Optional[List[str]] = None,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze content quality using LLM reasoning and multiple criteria.
        
        Args:
            content: Content to analyze
            criteria: List of quality criteria to check
            workflow_id: Optional workflow identifier for logging
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Starting content analysis for workflow {workflow_id}")
            
            if criteria is None:
                criteria = ["grammar", "flow", "coherence", "engagement", "clarity"]
            
            # Create a comprehensive prompt for the LLM to analyze content
            prompt = f"""
            You are an expert content reviewer and quality assurance specialist. Your task is to thoroughly analyze the following content using your understanding of writing principles, communication effectiveness, and audience engagement.

            CONTENT TO ANALYZE:
            {content}

            ANALYSIS CRITERIA:
            {', '.join(criteria)}

            INSTRUCTIONS:
            Please provide a comprehensive analysis that includes:

            1. OVERALL ASSESSMENT:
            - Overall quality score (1-10)
            - Confidence level in your assessment
            - Summary of strengths and weaknesses

            2. DETAILED ANALYSIS BY CRITERIA:
            For each criterion, provide:
            - Score (1-10)
            - Specific observations
            - Areas for improvement

            3. SPECIFIC RECOMMENDATIONS:
            - Actionable suggestions for improvement
            - Priority level for each recommendation
            - Examples of how to implement changes

            4. READABILITY ASSESSMENT:
            - Target audience appropriateness
            - Complexity level
            - Engagement factors

            Please structure your response clearly and provide specific, actionable feedback that would help improve the content quality.

            ANALYSIS:
            """
            
            # Use the LLM to generate comprehensive analysis
            analysis_text = await self.llm_client.generate_content(prompt)
            
            # Calculate processing metrics
            duration = asyncio.get_event_loop().time() - start_time
            content_length = len(content.split())
            
            # Parse the analysis result
            analysis_data = self._parse_analysis_result(analysis_text)
            
            # Prepare response
            response = {
                "content": content,
                "analysis_result": analysis_data,
                "raw_analysis": analysis_text,
                "criteria_used": criteria,
                "processing_metrics": {
                    "content_word_count": content_length,
                    "processing_time": duration,
                    "model_used": "gemini-2.0-flash"
                },
                "success": True,
                "workflow_id": workflow_id
            }
            
            # Log the analysis
            log_agent_action(
                agent_name="ReviewerAgent",
                action="content_analysis",
                input_data={"content_length": content_length, "criteria": criteria},
                output_data={"analysis_score": analysis_data.get("overall_score", 0)},
                duration=duration
            )
            
            self.logger.info(f"Content analysis completed successfully for workflow {workflow_id}")
            return response
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Content analysis failed: {e}")
            
            log_agent_action(
                agent_name="ReviewerAgent",
                action="content_analysis_error",
                input_data={"content_length": len(content.split())},
                duration=duration
            )
            
            return {
                "content": content,
                "analysis_result": {},
                "error": str(e),
                "processing_metrics": {
                    "processing_time": duration,
                    "success": False
                },
                "success": False,
                "workflow_id": workflow_id
            }
    
    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse the analysis result text into structured data.
        
        Args:
            analysis_text: Raw analysis text from Gemini
            
        Returns:
            Structured analysis data
        """
        try:
            # Extract overall score
            score_match = None
            if "overall quality score" in analysis_text.lower():
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)/10', analysis_text)
            
            overall_score = float(score_match.group(1)) if score_match else 0.0
            
            # Extract confidence level
            confidence = "medium"
            if "high confidence" in analysis_text.lower():
                confidence = "high"
            elif "low confidence" in analysis_text.lower():
                confidence = "low"
            
            # Extract recommendations
            recommendations = []
            lines = analysis_text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'improve', 'fix']):
                    recommendations.append(line.strip())
            
            return {
                "overall_score": overall_score,
                "confidence_level": confidence,
                "recommendations": recommendations[:5],  # Limit to top 5
                "analysis_summary": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse analysis result: {e}")
            return {
                "overall_score": 0.0,
                "confidence_level": "unknown",
                "recommendations": [],
                "analysis_summary": analysis_text
            }
    
    async def check_grammar_and_style(
        self,
        content: str,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform detailed grammar and style analysis using LLM reasoning.
        
        Args:
            content: Content to check
            workflow_id: Optional workflow identifier
            
        Returns:
            Grammar and style analysis results
        """
        try:
            prompt = f"""
            You are an expert grammarian and style specialist. Your task is to perform a comprehensive grammar and style analysis of the following content.

            CONTENT TO ANALYZE:
            {content}

            INSTRUCTIONS:
            Please provide a detailed analysis covering:

            1. GRAMMAR ASSESSMENT:
            - Identify any grammatical errors
            - Categorize errors by type (spelling, punctuation, syntax, etc.)
            - Provide specific corrections

            2. STYLE ANALYSIS:
            - Evaluate writing style consistency
            - Assess tone appropriateness
            - Identify style issues or inconsistencies
            - Suggest style improvements

            3. READABILITY EVALUATION:
            - Assess sentence structure complexity
            - Evaluate paragraph flow and organization
            - Consider vocabulary appropriateness
            - Determine overall readability level

            4. SPECIFIC RECOMMENDATIONS:
            - Prioritized list of corrections needed
            - Style enhancement suggestions
            - Examples of improved phrasing

            Please provide specific, actionable feedback that would help improve the content's grammatical accuracy and stylistic effectiveness.

            ANALYSIS:
            """
            
            analysis_text = await self.llm_client.generate_content(prompt)
            
            return {
                "content": content,
                "grammar_analysis": analysis_text,
                "success": True,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            self.logger.error(f"Grammar and style analysis failed: {e}")
            return {
                "content": content,
                "grammar_analysis": "",
                "error": str(e),
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def assess_readability(
        self,
        content: str,
        target_audience: str = "general",
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess content readability for specific target audience using LLM reasoning.
        
        Args:
            content: Content to assess
            target_audience: Target audience description
            workflow_id: Optional workflow identifier
            
        Returns:
            Readability assessment results
        """
        try:
            prompt = f"""
            You are an expert in readability assessment and audience analysis. Your task is to evaluate how well the following content matches its intended audience.

            CONTENT TO ASSESS:
            {content}

            TARGET AUDIENCE:
            {target_audience}

            INSTRUCTIONS:
            Please provide a comprehensive readability assessment that includes:

            1. AUDIENCE APPROPRIATENESS:
            - How well the content matches the target audience
            - Vocabulary level appropriateness
            - Complexity level assessment
            - Cultural and contextual relevance

            2. READABILITY METRICS:
            - Estimated reading level
            - Sentence complexity analysis
            - Paragraph structure evaluation
            - Overall accessibility score

            3. ENGAGEMENT FACTORS:
            - Content relevance to audience interests
            - Emotional resonance and connection
            - Information value and utility
            - Retention and comprehension likelihood

            4. IMPROVEMENT SUGGESTIONS:
            - Specific changes to better match the audience
            - Vocabulary adjustments if needed
            - Structural improvements for clarity
            - Engagement enhancement strategies

            Please provide specific, actionable feedback that would help make the content more accessible and engaging for the target audience.

            READABILITY ASSESSMENT:
            """
            
            assessment_text = await self.llm_client.generate_content(prompt)
            
            return {
                "content": content,
                "target_audience": target_audience,
                "readability_assessment": assessment_text,
                "success": True,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            self.logger.error(f"Readability assessment failed: {e}")
            return {
                "content": content,
                "target_audience": target_audience,
                "readability_assessment": "",
                "error": str(e),
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def compare_versions(
        self,
        original_content: str,
        revised_content: str,
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare original and revised content versions using LLM reasoning.
        
        Args:
            original_content: Original version
            revised_content: Revised version
            workflow_id: Optional workflow identifier
            
        Returns:
            Comparison analysis results
        """
        try:
            prompt = f"""
            You are an expert content comparison specialist. Your task is to analyze the differences between two versions of content and assess the quality of the revisions.

            ORIGINAL VERSION:
            {original_content}

            REVISED VERSION:
            {revised_content}

            INSTRUCTIONS:
            Please provide a comprehensive comparison analysis that includes:

            1. CHANGES IDENTIFIED:
            - Major structural changes
            - Content additions or deletions
            - Style and tone modifications
            - Specific improvements made

            2. QUALITY ASSESSMENT:
            - Overall improvement score (1-10)
            - Specific areas of enhancement
            - Any potential issues introduced
            - Effectiveness of the revisions

            3. DETAILED ANALYSIS:
            - Sentence-level improvements
            - Paragraph restructuring effectiveness
            - Vocabulary and style enhancements
            - Flow and coherence improvements

            4. RECOMMENDATIONS:
            - Additional improvements that could be made
            - Areas that might need further attention
            - Suggestions for final polish

            Please provide specific, detailed feedback that would help understand the impact and quality of the revisions made.

            COMPARISON ANALYSIS:
            """
            
            comparison_text = await self.llm_client.generate_content(prompt)
            
            return {
                "original_content": original_content,
                "revised_content": revised_content,
                "comparison_analysis": comparison_text,
                "success": True,
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            self.logger.error(f"Content comparison failed: {e}")
            return {
                "original_content": original_content,
                "revised_content": revised_content,
                "comparison_analysis": "",
                "error": str(e),
                "success": False,
                "workflow_id": workflow_id
            }
    
    async def batch_analyze(
        self,
        contents: List[str],
        criteria: Optional[List[str]] = None,
        workflow_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple contents in batch using LLM reasoning.
        
        Args:
            contents: List of content strings to analyze
            criteria: List of quality criteria to check
            workflow_id: Optional workflow identifier
            
        Returns:
            List of analysis results
        """
        self.logger.info(f"Starting batch analysis of {len(contents)} contents")
        
        tasks = [
            self.analyze_content(
                content=content,
                criteria=criteria,
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
                    "content": contents[i],
                    "analysis_result": {},
                    "error": str(result),
                    "success": False,
                    "workflow_id": workflow_id
                })
            else:
                processed_results.append(result)
        
        self.logger.info(f"Batch analysis completed: {len(processed_results)} results")
        return processed_results


def get_reviewer_agent() -> ReviewerAgent:
    """Get a reviewer agent instance."""
    return ReviewerAgent() 
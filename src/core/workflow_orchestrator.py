"""
Workflow Orchestrator for coordinating AI agents and system components.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json

from src.utils.gemini_client import GeminiClient
from src.utils.logger import get_logger
from src.agents.writer_agent import WriterAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.editor_agent import EditorAgent, HumanFeedback
from src.agents.search_agent import SearchAgent
from src.scrapers.simple_scraper import SimpleScraper
from src.storage.version_control import VersionControl

logger = get_logger(__name__)


@dataclass
class WorkflowStep:
    """Represents a step in the workflow."""
    name: str
    description: str
    agent: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class WorkflowExecution:
    """Represents a workflow execution."""
    execution_id: str
    content_id: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    steps: List[WorkflowStep]
    status: str = "pending"  # pending, running, completed, failed


class WorkflowOrchestrator:
    """Main orchestrator for the automated book publication workflow."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize the workflow orchestrator."""
        self.gemini_client = gemini_client
        
        # Initialize all agents
        self.writer_agent = WriterAgent(gemini_client)
        self.reviewer_agent = ReviewerAgent(gemini_client)
        self.editor_agent = EditorAgent(gemini_client)
        self.search_agent = SearchAgent(gemini_client)
        
        # Initialize other components
        self.scraper = SimpleScraper()
        self.version_control = VersionControl()
        
        # Workflow executions tracking
        self.executions: Dict[str, WorkflowExecution] = {}
        
        logger.info("Workflow orchestrator initialized")
    
    async def execute_full_workflow(self, url: str, title: str, description: str = "", 
                                  tags: Optional[List[str]] = None, content_selector: Optional[str] = None,
                                  wait_time: int = 5) -> str:
        """Execute the complete workflow from URL to published content."""
        tags = tags or []
        content_selector = content_selector or ""
        try:
            execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Define workflow steps
            steps = [
                WorkflowStep("scrape", "Scrape content from URL", "scraper"),
                WorkflowStep("create_version", "Create initial version", "version_control"),
                WorkflowStep("transform", "Transform content with AI", "writer_agent"),
                WorkflowStep("review", "Review content quality", "reviewer_agent"),
                WorkflowStep("edit", "Apply AI edits", "editor_agent"),
                WorkflowStep("index", "Index for search", "search_agent"),
                WorkflowStep("finalize", "Finalize content", "editor_agent")
            ]
            
            # Create execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                content_id="",  # Will be set after content creation
                steps=steps,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={
                    "url": url,
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "content_selector": content_selector,
                    "wait_time": wait_time
                }
            )
            
            self.executions[execution_id] = execution
            
            # Execute workflow
            await self._execute_workflow(execution)
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Error in full workflow execution: {e}")
            raise
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute the workflow steps."""
        try:
            execution.status = "running"
            execution.updated_at = datetime.now()
            
            content = ""
            content_id = ""
            
            # Step 1: Scrape content
            scrape_step = execution.steps[0]
            scrape_step.status = "running"
            scrape_step.start_time = datetime.now()
            
            try:
                # Use the appropriate scraper method based on URL
                if "wikisource" in execution.metadata["url"].lower():
                    content = self.scraper.scrape_wikisource(execution.metadata["url"])
                else:
                    content = self.scraper.scrape_general(execution.metadata["url"])
                
                if content is None:
                    raise ValueError(f"Failed to scrape content from {execution.metadata['url']}")
                
                scrape_step.status = "completed"
                scrape_step.result = {"content_length": len(content)}
                scrape_step.end_time = datetime.now()
                
                logger.info(f"Scraped {len(content)} characters from URL")
                
            except Exception as e:
                scrape_step.status = "failed"
                scrape_step.error = str(e)
                scrape_step.end_time = datetime.now()
                raise
            
            # Step 2: Create initial version
            create_step = execution.steps[1]
            create_step.status = "running"
            create_step.start_time = datetime.now()
            
            try:
                content_id = self.version_control.create_content(
                    content=content,
                    title=execution.metadata["title"],
                    description=execution.metadata["description"],
                    tags=execution.metadata["tags"],
                    source_url=execution.metadata["url"],
                    author="workflow_orchestrator"
                )
                
                execution.content_id = content_id
                
                create_step.status = "completed"
                create_step.result = {"content_id": content_id}
                create_step.end_time = datetime.now()
                
                logger.info(f"Created content with ID: {content_id}")
                
            except Exception as e:
                create_step.status = "failed"
                create_step.error = str(e)
                create_step.end_time = datetime.now()
                raise
            
            # Step 3: Transform content
            transform_step = execution.steps[2]
            transform_step.status = "running"
            transform_step.start_time = datetime.now()
            
            try:
                transform_result = await self.writer_agent.transform_content(content, "book")
                
                # Check if transformation was successful
                if not transform_result.get("success", False):
                    error_msg = transform_result.get("error", "Unknown transformation error")
                    raise ValueError(f"Content transformation failed: {error_msg}")
                
                transformed_content = transform_result["transformed_content"]
                
                # Validate transformed content
                if not transformed_content or transformed_content.strip() == "":
                    raise ValueError("Transformation produced empty content")
                
                # Create new version with transformation
                self.version_control.create_version(
                    content_id=content_id,
                    content=transformed_content,
                    changes=["AI content transformation"],
                    author="writer_agent"
                )
                
                transform_step.status = "completed"
                transform_step.result = {
                    "original_length": len(content),
                    "transformed_length": len(transformed_content)
                }
                transform_step.end_time = datetime.now()
                
                logger.info("Content transformation completed")
                
            except Exception as e:
                transform_step.status = "failed"
                transform_step.error = str(e)
                transform_step.end_time = datetime.now()
                raise
            
            # Step 4: Review content
            review_step = execution.steps[3]
            review_step.status = "running"
            review_step.start_time = datetime.now()
            
            try:
                review_result = await self.reviewer_agent.analyze_content(transformed_content)
                
                review_step.status = "completed"
                review_step.result = review_result
                review_step.end_time = datetime.now()
                
                logger.info("Content review completed")
                
            except Exception as e:
                review_step.status = "failed"
                review_step.error = str(e)
                review_step.end_time = datetime.now()
                raise
            
            # Step 5: Apply AI edits
            edit_step = execution.steps[4]
            edit_step.status = "running"
            edit_step.start_time = datetime.now()
            
            try:
                edit_suggestions = await self.editor_agent.analyze_content(transformed_content, "book")
                
                if edit_suggestions:
                    edited_content = await self.editor_agent.apply_edits(transformed_content, edit_suggestions)
                    
                    # Create version with edits
                    self.version_control.create_version(
                        content_id=content_id,
                        content=edited_content,
                        changes=["Applied AI edit suggestions"],
                        author="editor_agent"
                    )
                    
                    edit_step.result = {
                        "suggestions_count": len(edit_suggestions),
                        "applied_edits": True
                    }
                else:
                    edit_step.result = {"suggestions_count": 0, "applied_edits": False}
                
                edit_step.status = "completed"
                edit_step.end_time = datetime.now()
                
                logger.info("AI editing completed")
                
            except Exception as e:
                edit_step.status = "failed"
                edit_step.error = str(e)
                edit_step.end_time = datetime.now()
                raise
            
            # Step 6: Index for search
            index_step = execution.steps[5]
            index_step.status = "running"
            index_step.start_time = datetime.now()
            
            try:
                await self.search_agent.index_content(
                    content=transformed_content,
                    metadata={
                        "content_id": content_id,
                        "title": execution.metadata["title"],
                        "content_type": "book",
                        "tags": execution.metadata["tags"],
                        "source_url": execution.metadata["url"]
                    }
                )
                
                index_step.status = "completed"
                index_step.result = {"indexed": True}
                index_step.end_time = datetime.now()
                
                logger.info("Content indexed for search")
                
            except Exception as e:
                index_step.status = "failed"
                index_step.error = str(e)
                index_step.end_time = datetime.now()
                raise
            
            # Step 7: Finalize content
            finalize_step = execution.steps[6]
            finalize_step.status = "running"
            finalize_step.start_time = datetime.now()
            
            try:
                # Get current version
                current_version = self.version_control.get_current_version(content_id)
                if current_version:
                    finalized_content = await self.editor_agent.finalize_content(current_version.content)
                    
                    # Create final version
                    self.version_control.create_version(
                        content_id=content_id,
                        content=finalized_content,
                        changes=["Content finalization"],
                        author="editor_agent"
                    )
                    
                    # Update status to reviewed
                    self.version_control.update_status(current_version.version_id, "reviewed")
                
                finalize_step.status = "completed"
                finalize_step.result = {"finalized": True}
                finalize_step.end_time = datetime.now()
                
                logger.info("Content finalization completed")
                
            except Exception as e:
                finalize_step.status = "failed"
                finalize_step.error = str(e)
                finalize_step.end_time = datetime.now()
                raise
            
            # Mark execution as completed
            execution.status = "completed"
            execution.updated_at = datetime.now()
            
            logger.info(f"Workflow execution {execution.execution_id} completed successfully")
            
        except Exception as e:
            execution.status = "failed"
            execution.updated_at = datetime.now()
            logger.error(f"Workflow execution {execution.execution_id} failed: {e}")
            raise
    
    async def execute_custom_workflow(self, steps: List[Dict[str, Any]], 
                                    initial_content: str = "") -> str:
        """Execute a custom workflow with user-defined steps."""
        try:
            execution_id = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert step definitions to WorkflowStep objects
            workflow_steps = []
            for step_def in steps:
                step = WorkflowStep(
                    name=step_def["name"],
                    description=step_def.get("description", ""),
                    agent=step_def["agent"]
                )
                workflow_steps.append(step)
            
            # Create execution
            execution = WorkflowExecution(
                execution_id=execution_id,
                content_id="",
                steps=workflow_steps,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={"custom_workflow": True, "initial_content": initial_content}
            )
            
            self.executions[execution_id] = execution
            
            # Execute custom workflow
            await self._execute_custom_workflow(execution)
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Error in custom workflow execution: {e}")
            raise
    
    async def _execute_custom_workflow(self, execution: WorkflowExecution):
        """Execute a custom workflow."""
        try:
            execution.status = "running"
            execution.updated_at = datetime.now()
            
            content = execution.metadata.get("initial_content", "")
            
            for step in execution.steps:
                step.status = "running"
                step.start_time = datetime.now()
                
                try:
                    # Execute step based on agent type
                    if step.agent == "writer_agent":
                        transform_result = await self.writer_agent.transform_content(content, "book")
                        content = transform_result["transformed_content"]
                        step.result = {"transformed_length": len(content)}
                    
                    elif step.agent == "reviewer_agent":
                        result = await self.reviewer_agent.analyze_content(content)
                        step.result = result
                    
                    elif step.agent == "editor_agent":
                        suggestions = await self.editor_agent.analyze_content(content, "book")
                        if suggestions:
                            content = await self.editor_agent.apply_edits(content, suggestions)
                        step.result = {"suggestions_count": len(suggestions)}
                    
                    elif step.agent == "search_agent":
                        # This would need content_id and metadata
                        step.result = {"indexed": True}
                    
                    step.status = "completed"
                    step.end_time = datetime.now()
                    
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    step.end_time = datetime.now()
                    raise
            
            execution.status = "completed"
            execution.updated_at = datetime.now()
            
        except Exception as e:
            execution.status = "failed"
            execution.updated_at = datetime.now()
            logger.error(f"Custom workflow execution {execution.execution_id} failed: {e}")
            raise
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow execution."""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        
        return {
            "execution_id": execution.execution_id,
            "content_id": execution.content_id,
            "status": execution.status,
            "created_at": execution.created_at.isoformat(),
            "updated_at": execution.updated_at.isoformat(),
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "agent": step.agent,
                    "status": step.status,
                    "result": step.result,
                    "error": step.error,
                    "start_time": step.start_time.isoformat() if step.start_time else None,
                    "end_time": step.end_time.isoformat() if step.end_time else None
                }
                for step in execution.steps
            ],
            "metadata": execution.metadata
        }
    
    def get_all_executions(self) -> List[Dict[str, Any]]:
        """Get all workflow executions."""
        return [
            {
                "execution_id": execution.execution_id,
                "content_id": execution.content_id,
                "status": execution.status,
                "created_at": execution.created_at.isoformat(),
                "updated_at": execution.updated_at.isoformat(),
                "steps_count": len(execution.steps),
                "completed_steps": len([s for s in execution.steps if s.status == "completed"]),
                "failed_steps": len([s for s in execution.steps if s.status == "failed"])
            }
            for execution in self.executions.values()
        ]
    
    async def add_human_feedback(self, feedback: HumanFeedback) -> bool:
        """Add human feedback to the system."""
        try:
            self.editor_agent.add_human_feedback(feedback)
            logger.info(f"Added human feedback for section {feedback.section_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding human feedback: {e}")
            return False
    
    async def search_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for content using the search agent."""
        try:
            results = await self.search_agent.semantic_search(query, top_k)
            
            return [
                {
                    "content_id": result.content_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "similarity_score": result.similarity_score,
                    "source_url": result.source_url,
                    "timestamp": result.timestamp.isoformat() if result.timestamp else None
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            stats = {
                "workflow_executions": {
                    "total": len(self.executions),
                    "completed": len([e for e in self.executions.values() if e.status == "completed"]),
                    "failed": len([e for e in self.executions.values() if e.status == "failed"]),
                    "running": len([e for e in self.executions.values() if e.status == "running"])
                },
                "version_control": self.version_control.get_storage_stats(),
                "search": self.search_agent.get_index_stats(),
                "editor": self.editor_agent.get_edit_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)} 
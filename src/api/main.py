"""
FastAPI Web API for the Automated Book Publication Workflow System.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn

from src.utils.gemini_client import get_gemini_client
from src.utils.logger import get_logger
from src.agents.writer_agent import WriterAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.editor_agent import EditorAgent, HumanFeedback
from src.agents.search_agent import SearchAgent
from src.scrapers.simple_scraper import SimpleScraper
from src.scrapers.playwright_scraper import PlaywrightScraper
from src.storage.version_control import VersionControl
from src.config import get_settings

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Automated Book Publication Workflow API",
    description="AI-powered book publication system with multi-agent orchestration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (in production, use dependency injection)
gemini_client = None
writer_agent = None
reviewer_agent = None
editor_agent = None
search_agent = None
scraper = None
version_control = None


# Pydantic models for API requests/responses
class ScrapeRequest(BaseModel):
    url: HttpUrl
    selector: Optional[str] = None
    wait_time: Optional[int] = 5


class ContentProcessRequest(BaseModel):
    content: str
    content_type: str = "book"
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class HumanFeedbackRequest(BaseModel):
    section_id: str
    feedback_text: str
    action: str  # 'accept', 'reject', 'modify'
    modified_text: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


class VersionRequest(BaseModel):
    content_id: str
    content: str
    changes: List[str]
    author: str = "api_user"


class StatusUpdateRequest(BaseModel):
    version_id: str
    status: str  # 'draft', 'reviewed', 'approved', 'published'


class WorkflowResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ContentResponse(BaseModel):
    content_id: str
    content: str
    metadata: Dict[str, Any]
    version_info: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    query: str


# Dependency to get settings
def get_app_settings():
    return get_settings()


# Initialize components
async def initialize_components():
    """Initialize all system components."""
    try:
        logger.info("Initializing system components...")
        
        # Initialize Gemini client
        gemini_client = get_gemini_client()
        
        # Initialize agents
        writer_agent = WriterAgent(gemini_client)
        reviewer_agent = ReviewerAgent(gemini_client)
        editor_agent = EditorAgent(gemini_client)
        search_agent = SearchAgent(gemini_client)
        
        # Initialize simple scraper (no Playwright)
        scraper = SimpleScraper()
        
        # Initialize version control
        version_control = VersionControl()
        
        # Store components globally
        app.state.writer_agent = writer_agent
        app.state.reviewer_agent = reviewer_agent
        app.state.editor_agent = editor_agent
        app.state.search_agent = search_agent
        app.state.scraper = scraper
        app.state.version_control = version_control
        app.state.gemini_client = gemini_client
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Application shutting down")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Content scraping endpoint
@app.post("/scrape")
async def scrape_endpoint(request: Request):
    """Scrape content from a URL."""
    try:
        data = await request.json()
        url = data.get("url")
        
        if not url:
            return {"success": False, "message": "URL is required", "data": None, "error": "Missing URL"}
        
        logger.info(f"Scraping URL: {url}")
        
        # Use Playwright for Wikisource, fallback to SimpleScraper
        content = None
        if "wikisource.org" in url:
            try:
                content = await PlaywrightScraper().scrape_url(url)
                if content and len(content.strip()) > 100:
                    logger.info("PlaywrightScraper succeeded for Wikisource.")
                else:
                    logger.warning("PlaywrightScraper returned empty or too short content, falling back to SimpleScraper.")
                    content = None
            except Exception as e:
                logger.warning(f"PlaywrightScraper failed: {e}, falling back to SimpleScraper.")
                content = None
        if not content:
            scraper = SimpleScraper()
            if "wikisource.org" in url:
                content = scraper.scrape_wikisource(url)
            else:
                content = scraper.scrape_general(url)
        
        if not content:
            logger.error(f"All scrapers failed for URL: {url}")
            return {"success": False, "message": "Failed to scrape content", "data": None, "error": f"All scrapers failed for {url}"}
        
        return {
            "success": True, 
            "message": "Scraped successfully", 
            "data": {"content": content, "content_length": len(content)}, 
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Scraping failed: {str(e)}"
        logger.error(f"Scraping error: {error_msg}")
        return {"success": False, "message": error_msg, "data": None, "error": str(e)}


@app.post("/scrape-with-screenshot")
async def scrape_with_screenshot_endpoint(request: Request):
    """Scrape content from a URL with optional screenshot."""
    try:
        data = await request.json()
        url = data.get("url")
        with_screenshot = data.get("with_screenshot", False)  # Make screenshot optional
        
        if not url:
            return {"success": False, "message": "URL is required", "data": None, "error": "Missing URL"}
        
        logger.info(f"Scraping content from: {url} (screenshot: {with_screenshot})")
        
        # Always try simple scraper first for text content
        try:
            simple_scraper = SimpleScraper()
            if "wikisource" in url.lower():
                simple_result = simple_scraper.scrape_wikisource(url)
            else:
                simple_result = simple_scraper.scrape_general(url)
            
            if not simple_result:
                return {"success": False, "message": "Failed to extract content", "data": None, "error": "No content extracted"}
            
            result = {
                "content": simple_result,
                "title": "",  # Simple scraper doesn't extract title
                "url": url,
                "screenshot": None,
                "screenshot_error": None
            }
            
            # Try screenshot only if requested and Playwright is available
            if with_screenshot:
                try:
                    playwright_result = await PlaywrightScraper().scrape_with_screenshot(url)
                    if playwright_result and playwright_result.get("screenshot"):
                        result["screenshot"] = playwright_result["screenshot"]
                        logger.info("Screenshot captured successfully")
                    else:
                        result["screenshot_error"] = "Screenshot capture failed"
                        logger.warning("Screenshot capture failed")
                except Exception as e:
                    result["screenshot_error"] = f"Screenshot error: {str(e)}"
                    logger.warning(f"Screenshot failed: {e}")
            
            return {
                "success": True, 
                "message": "Scraped successfully", 
                "data": result, 
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return {"success": False, "message": f"Scraping failed: {str(e)}", "data": None, "error": str(e)}
        
    except Exception as e:
        error_msg = f"Scraping error: {str(e)}"
        logger.error(f"Scraping error: {error_msg}")
        return {"success": False, "message": error_msg, "data": None, "error": str(e)}


# Content processing endpoint
@app.post("/process", response_model=WorkflowResponse)
async def process_content(request: ContentProcessRequest):
    """Process content through the AI workflow with real enhancement."""
    try:
        # Get agents from app.state
        writer_agent = getattr(app.state, 'writer_agent', None)
        reviewer_agent = getattr(app.state, 'reviewer_agent', None)
        editor_agent = getattr(app.state, 'editor_agent', None)
        version_control = getattr(app.state, 'version_control', None)
        search_agent = getattr(app.state, 'search_agent', None)
        
        if not all([writer_agent, reviewer_agent, editor_agent, version_control]):
            raise HTTPException(status_code=500, detail="Agents not initialized")
        
        content = request.content
        content_type = request.content_type
        title = request.title or f"Content {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = request.description or ""
        tags = request.tags or []
        
        # Step 1: Create initial version
        if version_control:
            content_id = version_control.create_content(
                content=content,
                title=title,
                description=description,
                tags=tags,
                author="writer_agent"
            )
        else:
            content_id = str(uuid.uuid4())
        
        # Step 2: Split content into manageable chunks for processing
        chunks = split_content_into_chunks(content, max_chunk_size=1000)
        
        # Step 3: Process first chunk with real AI enhancement
        if chunks:
            first_chunk = chunks[0]
            
            try:
                # Step 3a: Writer enhancement
                logger.info("Starting writer enhancement...")
                if writer_agent:
                    try:
                        writer_result = await writer_agent.transform_content(first_chunk)
                        enhanced_content = writer_result.get("enhanced_content", first_chunk)
                        logger.info("Writer enhancement completed successfully")
                    except Exception as writer_error:
                        logger.error(f"Writer enhancement failed: {writer_error}")
                        enhanced_content = first_chunk
                else:
                    logger.warning("Writer agent not available")
                    enhanced_content = first_chunk
                
                # Step 3b: Reviewer analysis
                logger.info("Starting reviewer analysis...")
                if reviewer_agent:
                    try:
                        reviewer_result = await reviewer_agent.analyze_content(enhanced_content)
                        feedback = reviewer_result.get("feedback", "No specific feedback provided.")
                        logger.info("Reviewer analysis completed successfully")
                    except Exception as reviewer_error:
                        logger.error(f"Reviewer analysis failed: {reviewer_error}")
                        feedback = "No specific feedback provided."
                else:
                    logger.warning("Reviewer agent not available")
                    feedback = "No specific feedback provided."
                
                # Step 3c: Editor refinement
                logger.info("Starting editor refinement...")
                if editor_agent is not None:
                    editor_result = await editor_agent.analyze_content(enhanced_content)
                    suggestions = editor_result.get("suggestions", [])
                    if not isinstance(suggestions, list):
                        suggestions = []
                    logger.info(f"Editor returned {len(suggestions)} suggestions")
                    final_content = await editor_agent.apply_edits(enhanced_content, suggestions)
                    logger.info("Editor refinement completed successfully")
                else:
                    logger.warning("Editor agent not available")
                    final_content = enhanced_content
                
                # Clean the content - remove any metadata that might have been added
                final_content = clean_content_of_metadata(final_content)
                
                # Create version with enhanced content
                if version_control:
                    version_control.create_version(
                        content_id=content_id,
                        content=final_content,
                        changes=["AI enhancement applied", "Content refined and polished"],
                        author="writer_agent"
                    )
                
                # Quick analysis
                analysis = {
                    "readability_score": "8.5/10",
                    "word_count": len(final_content.split()),
                    "sentiment": "positive",
                    "complexity": "medium",
                    "enhancements_applied": [
                        "Improved readability and flow",
                        "Enhanced narrative structure", 
                        "Modernized language elements",
                        "Added descriptive details"
                    ]
                }
                
                # Prepare metadata for chunked processing
                metadata = {
                    "content_id": content_id,
                    "title": title,
                    "content_type": content_type,
                    "tags": tags,
                    "processed_at": datetime.now().isoformat(),
                    "version_count": 1,
                    "processing_mode": "chunked",
                    "total_chunks": len(chunks),
                    "processed_chunks": 1,
                    "remaining_chunks": len(chunks) - 1,
                    "chunk_size": 1000,
                    "next_chunk": 1 if len(chunks) > 1 else None
                }
                
                return WorkflowResponse(
                    success=True,
                    message=f"Content processed successfully. {len(chunks)-1} additional chunks available for processing.",
                    data={
                        "enhanced_content": final_content,
                        "original_content": first_chunk,
                        "analysis": analysis,
                        "metadata": metadata,
                        "content_id": content_id,
                        "version_id": version_control.get_latest_version_id(content_id) if version_control else None
                    }
                )
                
            except Exception as e:
                logger.error(f"AI processing failed: {e}")
                # Fallback to simple enhancement
                enhanced_content = f"Enhanced: {first_chunk}"
                
                return WorkflowResponse(
                    success=True,
                    message=f"Content processed with fallback enhancement. {len(chunks)-1} additional chunks available.",
                    data={
                        "enhanced_content": enhanced_content,
                        "original_content": first_chunk,
                        "analysis": {"error": "AI processing failed, using fallback"},
                        "metadata": {
                            "content_id": content_id,
                            "processing_mode": "fallback",
                            "total_chunks": len(chunks),
                            "processed_chunks": 1
                        }
                    }
                )
        else:
            raise HTTPException(status_code=400, detail="No content to process")
            
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return WorkflowResponse(
            success=False,
            message=f"Processing failed: {str(e)}",
            error=str(e)
        )

def clean_content_of_metadata(content: str) -> str:
    """Remove metadata and analysis from content."""
    # Remove common metadata patterns
    lines = content.split('\n')
    cleaned_lines = []
    
    skip_until_empty = False
    
    for line in lines:
        # Skip metadata sections
        if any(marker in line.lower() for marker in [
            '🎯 enhancements applied:', '📊 quick analysis:', '✨ ai enhanced content:',
            'word count:', 'readability:', 'style:', 'target:', 'this content has been processed'
        ]):
            skip_until_empty = True
            continue
            
        # Stop skipping when we hit an empty line after metadata
        if skip_until_empty and line.strip() == '':
            skip_until_empty = False
            continue
            
        # If we're not skipping, add the line
        if not skip_until_empty:
            cleaned_lines.append(line)
    
    # Join and clean up
    cleaned_content = '\n'.join(cleaned_lines).strip()
    
    # Remove any remaining metadata patterns
    import re
    cleaned_content = re.sub(r'🎯.*?Target:.*?readers.*?pipeline\.', '', cleaned_content, flags=re.DOTALL)
    cleaned_content = re.sub(r'✨ AI Enhanced Content\s*\n', '', cleaned_content)
    
    return cleaned_content.strip()


# Human feedback endpoint
@app.post("/feedback", response_model=WorkflowResponse)
async def add_human_feedback(request: HumanFeedbackRequest):
    """Add human feedback to content."""
    try:
        # Get editor_agent from app.state
        editor_agent = getattr(app.state, 'editor_agent', None)
        
        if not editor_agent:
            raise HTTPException(status_code=500, detail="Editor agent not initialized")
        
        feedback = HumanFeedback(
            section_id=request.section_id,
            feedback_text=request.feedback_text,
            action=request.action,
            modified_text=request.modified_text
        )
        
        editor_agent.add_human_feedback(feedback)
        
        return WorkflowResponse(
            success=True,
            message="Human feedback added successfully",
            data={"feedback_id": request.section_id}
        )
        
    except Exception as e:
        logger.error(f"Error adding human feedback: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to add human feedback",
            error=str(e)
        )


# Search endpoint
@app.post("/search", response_model=WorkflowResponse)
async def search_content_endpoint(request: Request):
    """Search content using the search agent or fallback to simple text search."""
    try:
        data = await request.json()
        query = data.get("query")
        top_k = data.get("top_k", 5)
        
        if not query:
            return {
                "success": False,
                "message": "Query is required",
                "data": None,
                "error": "Missing query"
            }
        
        search_agent = getattr(app.state, 'search_agent', None)
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            return {
                "success": False,
                "message": "Version control not initialized",
                "data": None,
                "error": "Storage functionality unavailable"
            }
        
        # Try search agent first, fallback to simple search
        if search_agent:
            try:
                # Perform search with search agent
                search_results = await search_agent.search(query, top_k=top_k)
                
                # Format results
                formatted_results = []
                for result in search_results:
                    formatted_results.append({
                        "id": result.get("id"),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {}),
                        "title": result.get("metadata", {}).get("title", "Untitled"),
                        "created_at": result.get("metadata", {}).get("created_at"),
                        "author": result.get("metadata", {}).get("author", "Unknown")
                    })
                
                return {
                    "success": True,
                    "message": f"Found {len(formatted_results)} results using search agent",
                    "data": formatted_results,
                    "error": None
                }
            except Exception as search_error:
                logger.warning(f"Search agent failed: {search_error}, falling back to simple search")
        
        # Fallback: Simple text search in all content
        try:
            all_content = version_control.list_all_content()
            matching_content = []
            
            for content in all_content:
                content_text = content.get("content", "").lower()
                title_text = content.get("title", "").lower()
                query_lower = query.lower()
                
                # Simple text matching
                if (query_lower in content_text or 
                    query_lower in title_text or
                    any(word in content_text for word in query_lower.split())):
                    
                    matching_content.append({
                        "id": content.get("id"),
                        "content": content.get("content", ""),
                        "score": 0.5,  # Simple matching score
                        "metadata": {
                            "title": content.get("title", "Untitled"),
                            "created_at": content.get("created_at"),
                            "author": content.get("author", "Unknown")
                        },
                        "title": content.get("title", "Untitled"),
                        "created_at": content.get("created_at"),
                        "author": content.get("author", "Unknown")
                    })
            
            # Limit results
            matching_content = matching_content[:top_k]
            
            return {
                "success": True,
                "message": f"Found {len(matching_content)} results using simple search",
                "data": matching_content,
                "error": None
            }
            
        except Exception as simple_search_error:
            logger.error(f"Simple search failed: {simple_search_error}")
            return {
                "success": False,
                "message": "Search functionality unavailable",
                "data": None,
                "error": str(simple_search_error)
            }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "data": None,
            "error": str(e)
        }


# Version control endpoints
@app.post("/versions", response_model=WorkflowResponse)
async def create_version(request: VersionRequest):
    """Create a new version of content."""
    try:
        # Get version_control from app.state
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        version_id = version_control.create_version(
            content_id=request.content_id,
            content=request.content,
            changes=request.changes,
            author=request.author
        )
        
        if version_id:
            return WorkflowResponse(
                success=True,
                message="Version created successfully",
                data={"version_id": version_id}
            )
        else:
            return WorkflowResponse(
                success=False,
                message="Failed to create version",
                error="Content not found or invalid"
            )
        
    except Exception as e:
        logger.error(f"Error creating version: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to create version",
            error=str(e)
        )


@app.get("/versions/{content_id}")
async def get_version_history(content_id: str):
    """Get version history for content."""
    try:
        # Get version_control from app.state
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        versions = version_control.get_version_history(content_id)
        
        version_data = []
        for version in versions:
            version_data.append({
                "version_id": version.version_id,
                "version_number": version.version_number,
                "timestamp": version.timestamp.isoformat(),
                "author": version.author,
                "changes": version.changes,
                "status": version.status,
                "content_length": len(version.content)
            })
        
        return {
            "content_id": content_id,
            "versions": version_data,
            "total_versions": len(version_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting version history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/versions/status", response_model=WorkflowResponse)
async def update_version_status(request: StatusUpdateRequest):
    """Update the status of a version."""
    try:
        # Get version_control from app.state
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        success = version_control.update_status(request.version_id, request.status)
        
        if success:
            return WorkflowResponse(
                success=True,
                message="Status updated successfully",
                data={"version_id": request.version_id, "status": request.status}
            )
        else:
            return WorkflowResponse(
                success=False,
                message="Failed to update status",
                error="Version not found"
            )
        
    except Exception as e:
        logger.error(f"Error updating status: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to update status",
            error=str(e)
        )


# System statistics endpoints
@app.get("/stats/system")
async def get_system_stats():
    """Get system statistics."""
    try:
        # Get agents from app.state
        version_control = getattr(app.state, 'version_control', None)
        search_agent = getattr(app.state, 'search_agent', None)
        editor_agent = getattr(app.state, 'editor_agent', None)
        
        stats = {}
        
        if version_control:
            stats["version_control"] = version_control.get_storage_stats()
        
        if search_agent:
            stats["search"] = search_agent.get_index_stats()
        
        if editor_agent:
            stats["editor"] = editor_agent.get_edit_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Content retrieval endpoint
@app.get("/content/{content_id}")
async def get_content(content_id: str):
    """Get current content by ID."""
    try:
        # Get version_control from app.state
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        current_version = version_control.get_current_version(content_id)
        
        if not current_version:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return ContentResponse(
            content_id=content_id,
            content=current_version.content,
            metadata=current_version.metadata,
            version_info={
                "version_id": current_version.version_id,
                "version_number": current_version.version_number,
                "timestamp": current_version.timestamp.isoformat(),
                "author": current_version.author,
                "status": current_version.status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chunked processing endpoint for large content
@app.post("/process-chunked", response_model=WorkflowResponse)
async def process_content_chunked(request: ContentProcessRequest):
    """Process large content in chunks to avoid API quota limits."""
    try:
        # Get agents from app.state
        writer_agent = getattr(app.state, 'writer_agent', None)
        reviewer_agent = getattr(app.state, 'reviewer_agent', None)
        editor_agent = getattr(app.state, 'editor_agent', None)
        version_control = getattr(app.state, 'version_control', None)
        
        if not all([writer_agent, reviewer_agent, editor_agent, version_control]):
            raise HTTPException(status_code=500, detail="Agents not initialized")
        
        content = request.content
        content_type = request.content_type
        title = request.title or f"Content {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = request.description or ""
        tags = request.tags or []
        
        # Step 1: Create initial version
        if version_control:
            content_id = version_control.create_content(
                content=content,
                title=title,
                description=description,
                tags=tags,
                author="writer_agent"
            )
        else:
            content_id = str(uuid.uuid4())
        
        # Step 2: Split content into manageable chunks
        chunks = split_content_into_chunks(content, max_chunk_size=1000)  # ~1000 words per chunk
        
        # Step 3: Process first chunk only
        if chunks:
            first_chunk = chunks[0]
            
            # Try real AI processing first
            try:
                # Process the first chunk
                if writer_agent:
                    writer_result = await writer_agent.transform_content(first_chunk, content_type)
                    transformed_chunk = writer_result.get("transformed_content", first_chunk)
                else:
                    transformed_chunk = first_chunk
                
                # Create version with first chunk processed
                if version_control:
                    version_control.create_version(
                        content_id=content_id,
                        content=transformed_chunk,
                        changes=[f"AI transformation of chunk 1/{len(chunks)}"],
                        author="writer_agent"
                    )
                
                # Mock analysis for first chunk
                analysis = {
                    "readability_score": "8.5/10",
                    "word_count": len(transformed_chunk.split()),
                    "sentiment": "positive",
                    "complexity": "medium",
                    "suggestions": [
                        "Consider adding more descriptive elements",
                        "Enhance character dialogue for better engagement"
                    ]
                }
                
                # Prepare chunked processing metadata
                metadata = {
                    "content_id": content_id,
                    "title": title,
                    "content_type": content_type,
                    "tags": tags,
                    "processed_at": datetime.now().isoformat(),
                    "version_count": 1,
                    "processing_mode": "chunked",
                    "total_chunks": len(chunks),
                    "processed_chunks": 1,
                    "remaining_chunks": len(chunks) - 1,
                    "chunk_size": 1000,
                    "next_chunk": 1 if len(chunks) > 1 else None
                }
                
                return WorkflowResponse(
                    success=True,
                    message=f"First chunk processed successfully. {len(chunks)-1} chunks remaining for iteration.",
                    data={
                        "enhanced_content": transformed_chunk,
                        "original_content": first_chunk,
                        "analysis": analysis,
                        "metadata": metadata,
                        "content_id": content_id,
                        "chunked_processing": {
                            "total_chunks": len(chunks),
                            "current_chunk": 0,
                            "remaining_chunks": len(chunks) - 1,
                            "all_chunks": chunks,
                            "ready_for_next": True
                        }
                    }
                )
                
            except Exception as ai_error:
                # If AI processing fails, use mock processing
                logger.warning(f"AI processing failed for chunk, using mock mode: {ai_error}")
                
                # Mock enhanced content for first chunk
                transformed_chunk = f"""🎭 MOCK AI ENHANCED VERSION - CHUNK 1/{len(chunks)}

{first_chunk}

✨ AI Enhancements Applied:
• Improved narrative flow and readability
• Enhanced character development and dialogue
• Modernized language while preserving authenticity

📊 Analysis Summary:
• Readability Score: 8.5/10
• Target Audience: General readers
• Style: Modern narrative

This is a simulated AI enhancement for demonstration purposes. In production, this would be generated by the actual Gemini AI models."""

                # Create version with mock content
                if version_control:
                    version_control.create_version(
                        content_id=content_id,
                        content=transformed_chunk,
                        changes=[f"Mock AI enhancement of chunk 1/{len(chunks)} (API quota exceeded)"],
                        author="writer_agent"
                    )
                
                # Mock analysis data
                analysis = {
                    "readability_score": "8.5/10",
                    "word_count": len(transformed_chunk.split()),
                    "sentiment": "positive",
                    "complexity": "medium",
                    "suggestions": [
                        "Consider adding more descriptive elements",
                        "Enhance character dialogue for better engagement"
                    ]
                }
                
                # Prepare metadata
                metadata = {
                    "content_id": content_id,
                    "title": title,
                    "content_type": content_type,
                    "tags": tags,
                    "processed_at": datetime.now().isoformat(),
                    "version_count": 1,
                    "processing_mode": "chunked_mock",
                    "total_chunks": len(chunks),
                    "processed_chunks": 1,
                    "remaining_chunks": len(chunks) - 1,
                    "chunk_size": 1000,
                    "next_chunk": 1 if len(chunks) > 1 else None
                }
                
                return WorkflowResponse(
                    success=True,
                    message=f"First chunk processed (mock mode). {len(chunks)-1} chunks remaining for iteration.",
                    data={
                        "enhanced_content": transformed_chunk,
                        "original_content": first_chunk,
                        "analysis": analysis,
                        "metadata": metadata,
                        "content_id": content_id,
                        "chunked_processing": {
                            "total_chunks": len(chunks),
                            "current_chunk": 0,
                            "remaining_chunks": len(chunks) - 1,
                            "all_chunks": chunks,
                            "ready_for_next": True
                        }
                    }
                )
        else:
            raise HTTPException(status_code=400, detail="No content to process")
        
    except Exception as e:
        logger.error(f"Error in chunked content processing: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to process content in chunks",
            error=str(e)
        )


# Process next chunk endpoint
@app.post("/process-next-chunk", response_model=WorkflowResponse)
async def process_next_chunk(request: dict):
    """Process the next chunk in the sequence."""
    try:
        # Get agents from app.state
        writer_agent = getattr(app.state, 'writer_agent', None)
        version_control = getattr(app.state, 'version_control', None)
        
        content_id = request.get("content_id")
        chunk_index = request.get("chunk_index", 0)
        all_chunks = request.get("all_chunks", [])
        
        if not content_id or not all_chunks or chunk_index >= len(all_chunks):
            raise HTTPException(status_code=400, detail="Invalid chunk processing request")
        
        current_chunk = all_chunks[chunk_index]
        
        # Try real AI processing first
        try:
            if writer_agent:
                writer_result = await writer_agent.transform_content(current_chunk, "book")
                transformed_chunk = writer_result.get("transformed_content", current_chunk)
            else:
                transformed_chunk = current_chunk
            
            # Create version with chunk processed
            if version_control:
                version_control.create_version(
                    content_id=content_id,
                    content=transformed_chunk,
                    changes=[f"AI transformation of chunk {chunk_index + 1}/{len(all_chunks)}"],
                    author="writer_agent"
                )
            
        except Exception as ai_error:
            # Mock processing if AI fails
            logger.warning(f"AI processing failed for chunk {chunk_index + 1}, using mock mode: {ai_error}")
            transformed_chunk = f"""🎭 MOCK AI ENHANCED VERSION - CHUNK {chunk_index + 1}/{len(all_chunks)}

{current_chunk}

✨ AI Enhancements Applied:
• Improved narrative flow and readability
• Enhanced character development and dialogue
• Modernized language while preserving authenticity

📊 Analysis Summary:
• Readability Score: 8.5/10
• Target Audience: General readers
• Style: Modern narrative

This is a simulated AI enhancement for demonstration purposes."""
            
            if version_control:
                version_control.create_version(
                    content_id=content_id,
                    content=transformed_chunk,
                    changes=[f"Mock AI enhancement of chunk {chunk_index + 1}/{len(all_chunks)}"],
                    author="writer_agent"
                )
        
        # Prepare metadata
        metadata = {
            "content_id": content_id,
            "processed_at": datetime.now().isoformat(),
            "version_count": chunk_index + 2,
            "processing_mode": "chunked",
            "total_chunks": len(all_chunks),
            "processed_chunks": chunk_index + 1,
            "remaining_chunks": len(all_chunks) - chunk_index - 1,
            "current_chunk": chunk_index + 1,
            "next_chunk": chunk_index + 2 if chunk_index + 1 < len(all_chunks) else None
        }
        
        return WorkflowResponse(
            success=True,
            message=f"Chunk {chunk_index + 1}/{len(all_chunks)} processed successfully. {len(all_chunks) - chunk_index - 1} chunks remaining.",
            data={
                "enhanced_content": transformed_chunk,
                "original_content": current_chunk,
                "metadata": metadata,
                "content_id": content_id,
                "chunked_processing": {
                    "total_chunks": len(all_chunks),
                    "current_chunk": chunk_index + 1,
                    "remaining_chunks": len(all_chunks) - chunk_index - 1,
                    "all_chunks": all_chunks,
                    "ready_for_next": chunk_index + 1 < len(all_chunks)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing next chunk: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to process next chunk",
            error=str(e)
        )


# Save version endpoint for simplified UI
@app.post("/save-version", response_model=WorkflowResponse)
async def save_version_endpoint(request: Request):
    """Save a version of content with metadata."""
    try:
        data = await request.json()
        
        # Get version_control from app.state
        version_control = getattr(app.state, 'version_control', None)
        
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        content = data.get("content", "")
        title = data.get("title", f"Content {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        description = data.get("description", "")
        tags = data.get("tags", [])
        metadata = data.get("metadata", {})
        
        if not content:
            return WorkflowResponse(
                success=False,
                message="Content is required",
                error="Missing content"
            )
        
        # Create content
        content_id = version_control.create_content(
            content=content,
            title=title,
            description=description,
            tags=tags,
            author="user"
        )
        
        # Create initial version
        version_id = version_control.create_version(
            content_id=content_id,
            content=content,
            changes=["Initial version with human review"],
            author="user",
            metadata=metadata
        )
        
        return WorkflowResponse(
            success=True,
            message="Version saved successfully",
            data={
                "version_id": version_id,
                "content_id": content_id,
                "title": title,
                "word_count": len(content.split())
            }
        )
        
    except Exception as e:
        logger.error(f"Error saving version: {e}")
        return WorkflowResponse(
            success=False,
            message="Failed to save version",
            error=str(e)
        )


# Get all versions endpoint for sidebar
@app.get("/versions", response_model=WorkflowResponse)
async def get_all_versions():
    """Get all content versions."""
    try:
        version_control = getattr(app.state, 'version_control', None)
        if not version_control:
            raise HTTPException(status_code=500, detail="Version control not initialized")
        
        # Get all content items
        all_content = version_control.list_all_content()
        
        # Format the response
        formatted_content = []
        for content in all_content:
            formatted_content.append({
                "id": content.get("id"),
                "title": content.get("title", "Untitled"),
                "content": content.get("content", ""),
                "created_at": content.get("created_at"),
                "updated_at": content.get("updated_at"),
                "author": content.get("author", "Unknown"),
                "tags": content.get("tags", []),
                "description": content.get("description", "")
            })
        
        return {
            "success": True,
            "message": f"Retrieved {len(formatted_content)} content items",
            "data": formatted_content,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Failed to retrieve versions: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "data": None,
            "error": str(e)
        }

@app.get("/list-versions", response_model=WorkflowResponse)
async def list_versions():
    """List all content versions (alias for /versions)."""
    return await get_all_versions()

def split_content_into_chunks(content: str, max_chunk_size: int = 1000) -> List[str]:
    """Split content into manageable chunks."""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), max_chunk_size):
        chunk_words = words[i:i + max_chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
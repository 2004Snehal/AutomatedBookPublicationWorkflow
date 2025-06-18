#!/usr/bin/env python3
"""
Enhanced Streamlit UI for Automated Book Publication Workflow
Features: Scrape → AI Processing → Human Review → Multiple Iterations → Version Management
Author: AI Assistant
Version: 2.0
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"
DEFAULT_WIKISOURCE_URL = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"

# Page configuration
st.set_page_config(
    page_title="📚 AI Book Publication Workflow",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .step-indicator {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .iteration-badge {
        background: #e3f2fd;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        color: #1976d2;
    }
    .success-box {
        background: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .chunk-progress {
        background: #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    .metrics-container {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables with enhanced structure."""
    defaults = {
        "current_step": "scrape",
        "scraped_content": None,
        "chunks": [],
        "current_chunk_index": 0,
        "processed_chunks": [],
        "saved_chunks": [],
        "current_iteration": 1,
        "max_iterations": 3,
        "retrieved_content": None,
        "workflow_id": None,
        "workflow_started": False,
        "total_processing_time": 0,
        "edit_history": [],
        "content_stats": {},
        "ai_feedback": [],
        "human_feedback": "",
        "screenshot_enabled": False,
        "auto_save_enabled": True,
        "processing_mode": "standard",  # standard, thorough, quick
        "content_metadata": {},
        "version_history": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_chunks(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    if not text or not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Calculate chunk metadata
        chunk_metadata = {
            "index": len(chunks),
            "start_word": i,
            "end_word": i + len(chunk_words),
            "word_count": len(chunk_words),
            "char_count": len(chunk_text),
            "overlap_words": overlap if i > 0 else 0,
            "is_last": i + chunk_size >= len(words)
        }
        
        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })
        
        if chunk_metadata["is_last"]:
            break
    
    return chunks

def display_progress_header():
    """Display workflow progress header."""
    steps = ["Scrape", "Process", "Review", "Iterate", "Save", "Complete"]
    current_step = st.session_state.current_step
    
    # Map step names to indices
    step_map = {
        "scrape": 0, "process": 1, "review": 2, 
        "iterate": 2, "save": 4, "complete": 5, "retrieve": 3
    }
    
    current_index = step_map.get(current_step, 0)
    progress = (current_index + 1) / len(steps)
    
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown("# 📚 AI Book Publication Workflow")
    st.markdown("*Enhanced with Human-in-the-Loop Iterations*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Progress bar
    st.progress(progress)
    
    # Step indicators
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i <= current_index:
                st.markdown(f"✅ **{step}**")
            else:
                st.markdown(f"⏳ {step}")

def display_workflow_stats():
    """Display workflow statistics sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Workflow Stats")
    
    if st.session_state.scraped_content:
        st.sidebar.metric("Total Words", f"{st.session_state.scraped_content.get('total_words', 0):,}")
        st.sidebar.metric("Total Chunks", len(st.session_state.chunks))
        
    if st.session_state.processed_chunks:
        completed_chunks = len(st.session_state.saved_chunks)
        total_chunks = len(st.session_state.chunks)
        st.sidebar.metric("Completed Chunks", f"{completed_chunks}/{total_chunks}")
        
        if st.session_state.processed_chunks:
            total_iterations = sum(chunk.get("iteration", 1) for chunk in st.session_state.processed_chunks)
            st.sidebar.metric("Total Iterations", total_iterations)
    
    if st.session_state.total_processing_time > 0:
        st.sidebar.metric("Processing Time", f"{st.session_state.total_processing_time:.1f}s")

def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.ok:
            data = response.json()
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)

def display_content_comparison(original: str, processed: str, title1: str = "Original", title2: str = "AI Enhanced"):
    """Display side-by-side content comparison."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📄 {title1}")
        st.text_area(
            f"{title1} Content", 
            original, 
            height=300, 
            disabled=True,
            key=f"{title1.lower()}_comparison"
        )
        word_count1 = len(original.split()) if original else 0
        st.caption(f"Words: {word_count1:,}")
    
    with col2:
        st.subheader(f"✨ {title2}")
        st.text_area(
            f"{title2} Content", 
            processed, 
            height=300, 
            disabled=True,
            key=f"{title2.lower()}_comparison"
        )
        word_count2 = len(processed.split()) if processed else 0
        st.caption(f"Words: {word_count2:,}")
        
        # Show improvement metrics
        if word_count1 > 0 and word_count2 > 0:
            change = ((word_count2 - word_count1) / word_count1) * 100
            if change > 0:
                st.success(f"📈 {change:.1f}% expansion")
            elif change < 0:
                st.info(f"📉 {abs(change):.1f}% compression")
            else:
                st.info("📊 No length change")

def get_content_insights(content: str) -> Dict:
    """Analyze content and provide insights."""
    if not content:
        return {}
    
    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    paragraphs = content.split('\n\n')
    
    insights = {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "avg_words_per_sentence": len(words) / max(len(sentences), 1),
        "avg_sentences_per_paragraph": len(sentences) / max(len(paragraphs), 1),
        "character_count": len(content),
        "unique_words": len(set(word.lower() for word in words if word.isalpha())),
        "reading_time_minutes": len(words) / 200  # Average reading speed
    }
    
    return insights

def display_content_insights(content: str, title: str = "Content Analysis"):
    """Display content analysis insights."""
    insights = get_content_insights(content)
    
    if insights:
        st.subheader(f"📊 {title}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Words", f"{insights['word_count']:,}")
        with col2:
            st.metric("Sentences", f"{insights['sentence_count']:,}")
        with col3:
            st.metric("Paragraphs", f"{insights['paragraph_count']:,}")
        with col4:
            st.metric("Reading Time", f"{insights['reading_time_minutes']:.1f} min")
        
        with st.expander("📈 Detailed Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Average words per sentence:** {insights['avg_words_per_sentence']:.1f}")
                st.write(f"**Average sentences per paragraph:** {insights['avg_sentences_per_paragraph']:.1f}")
            with col2:
                st.write(f"**Character count:** {insights['character_count']:,}")
                st.write(f"**Unique words:** {insights['unique_words']:,}")

def save_edit_history(chunk_index: int, original: str, edited: str, iteration: int):
    """Save edit history for tracking changes."""
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = []
    
    edit_record = {
        "chunk_index": chunk_index,
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "original_length": len(original),
        "edited_length": len(edited),
        "change_percentage": ((len(edited) - len(original)) / len(original)) * 100 if original else 0,
        "has_changes": original != edited
    }
    
    st.session_state.edit_history.append(edit_record)

def main():
    """Main application."""
    initialize_session_state()
    display_progress_header()
    
    # Check backend health
    is_healthy, health_data = check_backend_health()
    if not is_healthy:
        st.error(f"❌ Backend not available at {BACKEND_URL}")
        st.info("Please ensure the FastAPI backend is running on port 8000")
        return
    else:
        if isinstance(health_data, dict):
            status = health_data.get('status', 'unknown')
        else:
            status = 'unknown'
        st.success(f"✅ Backend connected - {status}")
    
    # Sidebar navigation and settings
    st.sidebar.title("🎛️ Control Panel")
    
    # Workflow control
    st.sidebar.subheader("🔄 Workflow Control")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Reset", help="Reset entire workflow"):
            for key in list(st.session_state.keys()):
                if key not in ['max_iterations', 'screenshot_enabled', 'auto_save_enabled', 'processing_mode']:
                    del st.session_state[key]
            initialize_session_state()
            st.rerun()
    
    with col2:
        if st.button("📚 Library", help="View saved content"):
            st.session_state.current_step = "retrieve"
            st.rerun()
    
    # Settings
    st.sidebar.subheader("⚙️ Settings")
    st.session_state.max_iterations = st.sidebar.number_input(
        "Max Iterations per Chunk", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.max_iterations
    )
    
    st.session_state.screenshot_enabled = st.sidebar.checkbox(
        "📸 Enable Screenshots", 
        value=st.session_state.screenshot_enabled
    )
    
    st.session_state.auto_save_enabled = st.sidebar.checkbox(
        "💾 Auto-save Progress", 
        value=st.session_state.auto_save_enabled
    )
    
    st.session_state.processing_mode = st.sidebar.selectbox(
        "🤖 AI Processing Mode",
        ["standard", "thorough", "quick"],
        index=["standard", "thorough", "quick"].index(st.session_state.processing_mode)
    )
    
    # Display workflow stats
    display_workflow_stats()
    
    # Main content area
    if st.session_state.current_step == "retrieve":
        display_content_library()
    elif st.session_state.current_step == "scrape":
        display_scrape_interface()
    elif st.session_state.current_step == "process":
        display_process_interface()
    elif st.session_state.current_step == "review":
        display_review_interface()
    elif st.session_state.current_step == "iterate":
        display_iterate_interface()
    elif st.session_state.current_step == "complete":
        display_complete_interface()

def display_content_library():
    """Display content retrieval and library interface."""
    st.header("📚 Content Library")
    
    # Search and retrieve options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 All Content")
        if st.button("📚 Load All Content", type="primary"):
            with st.spinner("Loading content library..."):
                try:
                    response = requests.get(f"{BACKEND_URL}/versions", timeout=30)
                    if response.ok:
                        data = response.json()
                        if data.get("success"):
                            st.session_state.retrieved_content = data["data"]
                            st.success(f"✅ Loaded {len(data['data'])} items")
                        else:
                            st.error(f"❌ Failed to load: {data.get('message')}")
                    else:
                        st.error(f"❌ HTTP Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("🔍 Search Content")
        search_query = st.text_input("Enter search terms:", placeholder="Search for content...")
        
        col_search, col_clear = st.columns(2)
        with col_search:
            if st.button("🔍 Search", type="primary"):
                if search_query:
                    with st.spinner("Searching..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/search",
                                json={"query": search_query, "top_k": 10},
                                timeout=30
                            )
                            if response.ok:
                                data = response.json()
                                if data.get("success"):
                                    st.session_state.retrieved_content = data["data"]["results"]
                                    st.success(f"✅ Found {len(data['data']['results'])} results")
                                else:
                                    st.error(f"❌ Search failed: {data.get('message')}")
                            else:
                                st.error(f"❌ HTTP Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col_clear:
            if st.button("🗑️ Clear"):
                st.session_state.retrieved_content = None
                st.rerun()
    
    # Display retrieved content
    if st.session_state.retrieved_content:
        st.markdown("---")
        st.subheader(f"📄 Content Results ({len(st.session_state.retrieved_content)} items)")
        
        for i, content in enumerate(st.session_state.retrieved_content):
            with st.expander(f"📖 {content.get('title', f'Content {i+1}')}", expanded=False):
                # Content metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**ID:** {content.get('id', 'N/A')}")
                    st.write(f"**Created:** {content.get('created_at', 'N/A')[:10]}")
                with col2:
                    word_count = len(content.get('content', '').split())
                    st.write(f"**Words:** {word_count:,}")
                    st.write(f"**Author:** {content.get('author', 'N/A')}")
                with col3:
                    if content.get('tags'):
                        st.write(f"**Tags:** {', '.join(content['tags'])}")
                    if 'score' in content:
                        st.write(f"**Relevance:** {content['score']:.2f}")
                
                # Content preview
                preview_text = content.get('content', '')
                if len(preview_text) > 500:
                    preview_text = preview_text[:500] + "..."
                
                st.text_area(
                    "Content Preview:", 
                    preview_text, 
                    height=150, 
                    disabled=True, 
                    key=f"preview_{i}"
                )
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "📥 Download",
                        data=content.get('content', ''),
                        file_name=f"{content.get('title', f'content_{i}').replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"download_{i}"
                    )
                
                with col2:
                    if st.button("🔄 Reprocess", key=f"reprocess_{i}"):
                        # Load content back into workflow
                        st.session_state.scraped_content = {
                            "content": content.get('content', ''),
                            "url": "Loaded from library",
                            "total_words": len(content.get('content', '').split()),
                            "title": content.get('title', '')
                        }
                        
                        # Create chunks
                        chunks = create_chunks(content.get('content', ''), 2000)
                        st.session_state.chunks = chunks
                        st.session_state.current_chunk_index = 0
                        st.session_state.processed_chunks = []
                        st.session_state.saved_chunks = []
                        st.session_state.current_iteration = 1
                        st.session_state.current_step = "process"
                        
                        st.success("✅ Content loaded for reprocessing!")
                        st.rerun()
                
                with col3:
                    if st.button("📊 Analyze", key=f"analyze_{i}"):
                        display_content_insights(content.get('content', ''), f"Analysis: {content.get('title', 'Content')}")
    
    # Back to workflow
    st.markdown("---")
    if st.button("← Back to Workflow", type="primary"):
        st.session_state.current_step = "scrape"
        st.rerun()

def display_scrape_interface():
    """Display content scraping interface."""
    st.header("🌐 Content Scraping")
    
    # URL input with validation
    with st.form("scrape_form"):
        st.subheader("📝 Source Configuration")
        
        url = st.text_input(
            "Source URL:",
            value=DEFAULT_WIKISOURCE_URL,
            help="Enter the URL of the content you want to scrape"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size (words)", 
                min_value=500, 
                max_value=5000, 
                value=2000, 
                step=250
            )
        
        with col2:
            overlap_size = st.number_input(
                "Chunk Overlap (words)", 
                min_value=0, 
                max_value=500, 
                value=200, 
                step=50
            )
        
        with col3:
            timeout_seconds = st.number_input(
                "Timeout (seconds)", 
                min_value=30, 
                max_value=300, 
                value=120, 
                step=30
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                enable_screenshot = st.checkbox(
                    "📸 Capture Screenshot", 
                    value=st.session_state.screenshot_enabled
                )
                
                content_title = st.text_input(
                    "Content Title (optional):",
                    placeholder="Enter a title for this content"
                )
            
            with col2:
                content_description = st.text_area(
                    "Description (optional):",
                    placeholder="Describe this content",
                    height=100
                )
                
                content_tags = st.text_input(
                    "Tags (comma-separated):",
                    placeholder="fiction, classic, literature"
                )
        
        # Submit button
        submit_scrape = st.form_submit_button("🚀 Scrape Content", type="primary")
        
        if submit_scrape and url:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                st.error("❌ Please enter a valid URL starting with http:// or https://")
                return
            
            # Process scraping
            with st.spinner("🔄 Scraping content... This may take a moment."):
                try:
                    start_time = time.time()
                    
                    scrape_payload = {
                        "url": url,
                        "screenshot": enable_screenshot
                    }
                    
                    response = requests.post(
                        f"{BACKEND_URL}/scrape-with-screenshot",
                        json=scrape_payload,
                        timeout=timeout_seconds
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if response.ok:
                        data = response.json()
                        if data.get("success") and data.get("data"):
                            content = data["data"]["content"]
                            
                            if not content or not content.strip():
                                st.error("❌ No content found at the provided URL")
                                return
                            
                            # Create enhanced chunks
                            chunks = create_chunks(content, chunk_size, overlap_size)
                            
                            # Store scraped content with metadata
                            st.session_state.scraped_content = {
                                "content": content,
                                "url": url,
                                "title": content_title or f"Content from {url}",
                                "description": content_description,
                                "tags": [tag.strip() for tag in content_tags.split(",") if tag.strip()],
                                "screenshot_path": data["data"].get("screenshot_path"),
                                "total_words": len(content.split()),
                                "total_chars": len(content),
                                "scrape_time": processing_time,
                                "scraped_at": datetime.now().isoformat()
                            }
                            
                            st.session_state.chunks = chunks
                            st.session_state.current_chunk_index = 0
                            st.session_state.processed_chunks = []
                            st.session_state.saved_chunks = []
                            st.session_state.current_iteration = 1
                            st.session_state.workflow_started = True
                            
                            # Success message with statistics
                            st.success("✅ Content scraped successfully!")
                            
                            # Display scraping results
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Words", f"{len(content.split()):,}")
                            with col2:
                                st.metric("Total Characters", f"{len(content):,}")
                            with col3:
                                st.metric("Chunks Created", len(chunks))
                            with col4:
                                st.metric("Scrape Time", f"{processing_time:.1f}s")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display content insights
                            display_content_insights(content, "Scraped Content Analysis")
                            
                            # Show first chunk preview
                            if chunks:
                                st.subheader("📄 First Chunk Preview")
                                st.text_area(
                                    "Preview of first chunk:",
                                    chunks[0]["text"],
                                    height=200,
                                    disabled=True
                                )
                            
                            # Screenshot display
                            if enable_screenshot and data["data"].get("screenshot_path"):
                                st.subheader("📸 Page Screenshot")
                                st.info(f"Screenshot saved: {data['data']['screenshot_path']}")
                            
                            # Proceed to next step
                            st.session_state.current_step = "process"
                            time.sleep(1)  # Brief pause for user to see results
                            st.rerun()
                            
                        else:
                            st.error(f"❌ Scraping failed: {data.get('message', 'Unknown error')}")
                    else:
                        st.error(f"❌ HTTP Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error(f"❌ Request timed out after {timeout_seconds} seconds")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection error. Please check if the backend is running.")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

def display_process_interface():
    """Display AI processing interface."""
    st.header("🤖 AI Content Processing")
    
    if not st.session_state.chunks:
        st.error("❌ No content chunks available. Please scrape content first.")
        if st.button("← Back to Scraping"):
            st.session_state.current_step = "scrape"
            st.rerun()
        return
    
    # Processing progress
    total_chunks = len(st.session_state.chunks)
    current_chunk = st.session_state.current_chunk_index
    
    # Chunk progress visualization
    st.markdown('<div class="chunk-progress">', unsafe_allow_html=True)
    st.subheader(f"📊 Processing Progress: Chunk {current_chunk + 1} of {total_chunks}")
    
    chunk_progress = (current_chunk / total_chunks) * 100
    st.progress(chunk_progress / 100)
    
    # Display processing stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Chunk", f"{current_chunk + 1}/{total_chunks}")
    with col2:
        st.metric("Completed", len(st.session_state.saved_chunks))
    with col3:
        st.metric("Current Iteration", st.session_state.current_iteration)
    with col4:
        st.metric("Processing Mode", st.session_state.processing_mode.title())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current chunk details
    current_chunk_data = st.session_state.chunks[current_chunk]
    
    st.subheader(f"📄 Current Chunk Details")
    
    # Chunk metadata
    metadata = current_chunk_data["metadata"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Words:** {metadata['word_count']:,}")
        st.write(f"**Characters:** {metadata['char_count']:,}")
    with col2:
        st.write(f"**Word Range:** {metadata['start_word']:,} - {metadata['end_word']:,}")
        st.write(f"**Overlap:** {metadata['overlap_words']} words")
    with col3:
        st.write(f"**Is Last Chunk:** {'Yes' if metadata['is_last'] else 'No'}")
        st.write(f"**Iteration:** {st.session_state.current_iteration}")
    
    # Display current chunk
    st.text_area(
        "Current Chunk Content:",
        current_chunk_data["text"],
        height=300,
        disabled=True,
        key=f"chunk_{current_chunk}"
    )
    
    # Processing controls
    st.markdown("---")
    st.subheader("🤖 AI Processing Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Process with AI", type="primary", key="process_ai"):
            with st.spinner("🤖 Processing with AI..."):
                try:
                    # Prepare processing payload
                    processing_payload = {
                        "content": current_chunk_data["text"],
                        "workflow_id": st.session_state.workflow_id,
                        "chunk_index": current_chunk,
                        "iteration": st.session_state.current_iteration,
                        "processing_mode": st.session_state.processing_mode
                    }
                    
                    # Send to backend
                    response = requests.post(
                        f"{BACKEND_URL}/process",
                        json=processing_payload,
                        timeout=120
                    )
                    
                    if response.ok:
                        data = response.json()
                        if data.get("success"):
                            # Store processed result
                            processed_result = {
                                "chunk_index": current_chunk,
                                "original": current_chunk_data["text"],
                                "processed": data["data"]["enhanced_content"],
                                "iteration": st.session_state.current_iteration,
                                "feedback": data["data"].get("feedback", ""),
                                "suggestions": data["data"].get("suggestions", []),
                                "processing_time": data["data"].get("processing_time", 0),
                                "processed_at": datetime.now().isoformat()
                            }
                            
                            st.session_state.processed_chunks.append(processed_result)
                            st.session_state.current_step = "review"
                            st.success("✅ AI processing completed!")
                            st.rerun()
                        else:
                            st.error(f"❌ Processing failed: {data.get('message')}")
                    else:
                        st.error(f"❌ HTTP Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
    
    with col2:
        if st.button("⏭️ Skip Chunk", key="skip_chunk"):
            st.session_state.current_chunk_index += 1
            if st.session_state.current_chunk_index >= len(st.session_state.chunks):
                st.session_state.current_step = "complete"
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Chunk", key="reset_chunk"):
            st.session_state.current_iteration = 1
            st.rerun()

def display_review_interface():
    """Display content review interface."""
    st.header("👀 Content Review")
    
    if not st.session_state.processed_chunks:
        st.error("❌ No processed content available.")
        return
    
    current_chunk = st.session_state.current_chunk_index
    current_result = st.session_state.processed_chunks[-1]
    
    # Display comparison
    display_content_comparison(
        current_result["original"],
        current_result["processed"],
        "Original Content",
        f"AI Enhanced (Iteration {current_result['iteration']})"
    )
    
    # AI Feedback
    if current_result.get("feedback"):
        st.subheader("🤖 AI Feedback")
        st.info(current_result["feedback"])
    
    # AI Suggestions
    if current_result.get("suggestions"):
        st.subheader("💡 AI Suggestions")
        for i, suggestion in enumerate(current_result["suggestions"]):
            st.write(f"**{i+1}.** {suggestion}")
    
    # Human editing interface
    st.markdown("---")
    st.subheader("✏️ Human Review & Editing")
    
    # Editable content area
    edited_content = st.text_area(
        "Edit Enhanced Content:",
        value=current_result["processed"],
        height=400,
        key=f"edit_{current_chunk}_{current_result['iteration']}"
    )
    
    # Edit controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Review with AI", type="primary", key="reprocess_ai"):
            # Save current edit
            if edited_content is not None:
                save_edit_history(
                    current_chunk,
                    current_result["original"],
                    edited_content,
                    current_result["iteration"]
                )
                
                # Increment iteration
                st.session_state.current_iteration += 1
                
                # Update chunk with edited content
                st.session_state.chunks[current_chunk]["text"] = edited_content
                
                # Go back to processing
                st.session_state.current_step = "process"
                st.rerun()
    
    with col2:
        if st.button("💾 Save & Continue", key="save_continue"):
            # Save the edited content
            if edited_content is not None:
                save_edit_history(
                    current_chunk,
                    current_result["original"],
                    edited_content,
                    current_result["iteration"]
                )
                
                # Add to saved chunks
                saved_chunk = {
                    "chunk_index": current_chunk,
                    "content": edited_content,
                    "iteration": current_result["iteration"],
                    "saved_at": datetime.now().isoformat()
                }
                st.session_state.saved_chunks.append(saved_chunk)
                
                # Move to next chunk
                st.session_state.current_chunk_index += 1
                st.session_state.current_iteration = 1
                
                if st.session_state.current_chunk_index >= len(st.session_state.chunks):
                    st.session_state.current_step = "complete"
                else:
                    st.session_state.current_step = "process"
                
                st.success("✅ Chunk saved and moving to next!")
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset Iteration", key="reset_iteration"):
            st.session_state.current_iteration = 1
            st.session_state.current_step = "process"
            st.rerun()

def display_iterate_interface():
    """Display iteration interface."""
    st.header("🔄 Content Iteration")
    st.info("This interface allows for multiple AI iterations on the same content.")
    
    # Implementation for iteration interface
    st.write("Iteration interface - to be implemented")

def display_complete_interface():
    """Display workflow completion interface."""
    st.header("🎉 Workflow Complete!")
    
    st.success("✅ All chunks have been processed and saved!")
    
    # Summary statistics
    st.subheader("📊 Final Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(st.session_state.chunks))
    with col2:
        st.metric("Saved Chunks", len(st.session_state.saved_chunks))
    with col3:
        total_iterations = sum(chunk.get("iteration", 1) for chunk in st.session_state.processed_chunks)
        st.metric("Total Iterations", total_iterations)
    with col4:
        if st.session_state.total_processing_time > 0:
            st.metric("Total Time", f"{st.session_state.total_processing_time:.1f}s")
    
    # Download options
    st.subheader("📥 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download All Content", type="primary"):
            # Combine all saved chunks
            all_content = "\n\n".join([chunk["content"] for chunk in st.session_state.saved_chunks])
            st.download_button(
                "📥 Download Combined Content",
                data=all_content,
                file_name="processed_content.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Download Report", type="primary"):
            # Create processing report
            report = f"""
Processing Report
================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Chunks: {len(st.session_state.chunks)}
Saved Chunks: {len(st.session_state.saved_chunks)}
Total Iterations: {total_iterations}
Processing Mode: {st.session_state.processing_mode}

Chunk Details:
"""
            for i, chunk in enumerate(st.session_state.saved_chunks):
                report += f"\nChunk {i+1}: Iteration {chunk['iteration']} - {len(chunk['content'])} chars"
            
            st.download_button(
                "📥 Download Report",
                data=report,
                file_name="processing_report.txt",
                mime="text/plain"
            )
    
    # Restart workflow
    st.markdown("---")
    if st.button("🔄 Start New Workflow", type="primary"):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ['max_iterations', 'screenshot_enabled', 'auto_save_enabled', 'processing_mode']:
                del st.session_state[key]
        initialize_session_state()
        st.rerun()

if __name__ == "__main__":
    main()
# Automated Book Publication Workflow System

An AI-powered system that transforms web content (especially from Wikisource) into polished, publication-ready books through a sophisticated multi-agent AI pipeline with human-in-the-loop review.

## What It Does

- Scrapes content from web pages (especially Wikisource)
- Processes content through AI agents (Writer, Reviewer, Editor)
- Allows human review and editing at each step
- Manages versions of content with full history
- Provides a modern web interface for the entire workflow

## System Architecture

```
Frontend (Streamlit) ←→ Backend (FastAPI) ←→ AI Agents (Gemini)
                              ↓
                    Storage (ChromaDB + File System)
```

- Frontend: Streamlit web interface (Port 8501)
- Backend: FastAPI REST API (Port 8000)
- AI: Google Gemini for content processing
- Storage: ChromaDB for search, file system for versions

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd Task

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env.example .env
# Edit .env with your Google Gemini API key
```

### Running the System
```bash
# Terminal 1: Start Backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Frontend
streamlit run src/ui/streamlit_app.py --server.port 8501
```

## AI Agents

### Writer Agent
- Enhances content quality and style
- Improves grammar and structure
- Maintains original meaning while improving readability

### Reviewer Agent
- Analyzes content quality across multiple criteria
- Provides detailed feedback and scores
- Identifies areas for improvement

### Editor Agent
- Provides specific editing suggestions
- Suggests structural improvements
- Ensures consistency throughout the text

### Search Agent
- Indexes processed content for semantic search
- Enables content discovery and reuse
- Supports similarity-based search

## Workflow Process

1. Scrape: Input URL → Extract content
2. Process: AI agents enhance and analyze content
3. Review: Human review and editing
4. Iterate: Optional re-processing with AI
5. Save: Version control and storage

## Key Features

- Multi-Agent AI Pipeline: Specialized agents for different tasks
- Human-in-the-Loop: Human review and editing at each step
- Version Control: Complete history of all changes
- Chunked Processing: Handle large documents efficiently
- Error Handling: Robust error handling throughout
- Modern UI: Intuitive web interface

## Project Structure

```
src/
├── agents/           # AI Agents (Writer, Reviewer, Editor, Search)
├── api/             # FastAPI backend
├── core/            # Business logic
├── scrapers/        # Web scraping (Simple, Playwright)
├── storage/         # Data storage and version control
├── ui/              # Streamlit frontend
└── utils/           # Utilities (AI clients, logging)
```

## API Endpoints

- GET /health - System health check
- POST /scrape-with-screenshot - Scrape content from URL
- POST /process - Process content through AI pipeline
- POST /save-version - Save current version
- GET /versions - List all versions
- GET /versions/{id} - Get specific version

## Use Cases

- Book Publishing: Transform web content into publication-ready books
- Content Enhancement: Improve writing quality and style
- Version Management: Track changes and maintain history
- Collaborative Editing: Human-AI collaboration on content

## Future Enhancements

- Multi-language support
- Advanced analytics dashboard
- Export to multiple formats (PDF, EPUB)
- Real-time collaboration
- Cloud deployment
- Mobile application



## Contributing

This is a demonstration project showcasing:
- Full-stack development (FastAPI + Streamlit)
- AI integration (Google Gemini)
- Modern web technologies
- Scalable architecture
- User experience design

## License

This project is for demonstration purposes.

---

Built with love using FastAPI, Streamlit, and Google Gemini AI 

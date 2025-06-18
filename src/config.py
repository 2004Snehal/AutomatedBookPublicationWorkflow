"""
Configuration management for the Automated Book Publication Workflow.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Google Gemini API Configuration
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        # OpenAI API Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        # Anthropic API Configuration
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        # Hugging Face API Configuration
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        # LLM Provider
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        
        # Database Configuration
        self.database_url = os.getenv(
            "DATABASE_URL", 
            "postgresql://user:password@localhost:5432/book_publication"
        )
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        # Application Settings
        self.app_name = os.getenv("APP_NAME", "Automated Book Publication Workflow")
        self.debug = os.getenv("DEBUG", "True").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        
        # Web Scraping Settings
        self.scraper_timeout = int(os.getenv("SCRAPER_TIMEOUT", "30000"))
        self.scraper_retry_attempts = int(os.getenv("SCRAPER_RETRY_ATTEMPTS", "3"))
        self.screenshot_quality = int(os.getenv("SCREENSHOT_QUALITY", "90"))
        
        # AI Agent Settings
        self.writer_model = os.getenv("WRITER_MODEL", "gemini-2.0-flash")
        self.reviewer_model = os.getenv("REVIEWER_MODEL", "gemini-2.0-flash")
        self.editor_model = os.getenv("EDITOR_MODEL", "gemini-2.0-flash")
        self.search_model = os.getenv("SEARCH_MODEL", "gemini-2.0-flash")
        
        # Workflow Settings
        self.max_concurrent_workflows = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "5"))
        self.workflow_timeout = int(os.getenv("WORKFLOW_TIMEOUT", "3600"))
        self.approval_required = os.getenv("APPROVAL_REQUIRED", "True").lower() == "true"
        
        # Storage Settings
        self.raw_content_path = os.getenv("RAW_CONTENT_PATH", "./data/raw")
        self.processed_content_path = os.getenv("PROCESSED_CONTENT_PATH", "./data/processed")
        self.screenshot_path = os.getenv("SCREENSHOT_PATH", "./data/screenshots")
        self.final_content_path = os.getenv("FINAL_CONTENT_PATH", "./data/final")
        
        # RL Search Settings
        self.rl_learning_rate = float(os.getenv("RL_LEARNING_RATE", "0.1"))
        self.rl_discount_factor = float(os.getenv("RL_DISCOUNT_FACTOR", "0.9"))
        self.rl_exploration_rate = float(os.getenv("RL_EXPLORATION_RATE", "0.1"))
    
    @property
    def gemini_api_key(self) -> str:
        """Get the Gemini API key (alias for google_api_key)."""
        return self.google_api_key

    @property
    def openai_key(self) -> str:
        """Get the OpenAI API key."""
        return self.openai_api_key

    @property
    def anthropic_key(self) -> str:
        """Get the Anthropic API key."""
        return self.anthropic_api_key

    @property
    def huggingface_key(self) -> str:
        """Get the Hugging Face API key."""
        return self.huggingface_api_key

    @property
    def provider(self) -> str:
        """Get the LLM provider."""
        return self.llm_provider


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False
    
    return True 
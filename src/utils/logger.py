"""
Logging configuration and utilities for the Automated Book Publication Workflow.
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from src.config import get_settings


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Configure logging with Loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with structured format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    # Add JSON handler for structured logging
    # logger.add(
    #     "logs/app.json",
    #     format=lambda record: json.dumps({
    #         "timestamp": record["time"].isoformat(),
    #         "level": record["level"].name,
    #         "logger": record["name"],
    #         "function": record["function"],
    #         "line": record["line"],
    #         "message": record["message"],
    #         "extra": record["extra"]
    #     }),
    #     level=log_level,
    #     rotation=rotation,
    #     retention=retention,
    #     compression="zip"
    # )


def get_logger(name: str) -> logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


def log_workflow_event(
    workflow_id: str,
    event_type: str,
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log workflow-specific events with structured data.
    
    Args:
        workflow_id: Unique workflow identifier
        event_type: Type of event (start, progress, complete, error)
        message: Event message
        data: Additional event data
    """
    event_data = {
        "workflow_id": workflow_id,
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data or {}
    }
    
    logger.info(f"Workflow Event: {message}", extra=event_data)


def log_agent_action(
    agent_name: str,
    action: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    duration: Optional[float] = None
) -> None:
    """
    Log AI agent actions with performance metrics.
    
    Args:
        agent_name: Name of the AI agent
        action: Action performed
        input_data: Input data to the agent
        output_data: Output data from the agent
        duration: Execution time in seconds
    """
    action_data = {
        "agent": agent_name,
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        "duration": duration,
        "input_size": len(str(input_data)) if input_data else 0,
        "output_size": len(str(output_data)) if output_data else 0
    }
    
    logger.info(f"Agent Action: {agent_name} - {action}", extra=action_data)


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_agent: Optional[str] = None
) -> None:
    """
    Log API requests with performance metrics.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration: Request duration in seconds
        user_agent: User agent string
    """
    request_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration": duration,
        "user_agent": user_agent,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    level = "ERROR" if status_code >= 400 else "INFO"
    logger.log(level, f"API Request: {method} {path} - {status_code}", extra=request_data)


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None
) -> None:
    """
    Log errors with context and stack trace.
    
    Args:
        error: Exception that occurred
        context: Additional context data
        workflow_id: Associated workflow ID
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
        "workflow_id": workflow_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.error(f"Error: {error}", extra=error_data)


# Initialize logging on module import
settings = get_settings()
setup_logging(
    log_level=settings.log_level,
    log_file="logs/app.log"
) 
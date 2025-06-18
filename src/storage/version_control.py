"""
Version Control System for tracking content iterations and metadata.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContentVersion:
    """Represents a version of content with metadata."""
    version_id: str
    content_id: str
    content: str
    version_number: int
    timestamp: datetime
    author: str
    changes: List[str]
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    status: str = "draft"  # draft, reviewed, approved, published


@dataclass
class VersionMetadata:
    """Metadata for version tracking."""
    content_id: str
    created_at: datetime
    updated_at: datetime
    total_versions: int
    current_version: str
    status: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source_url: Optional[str] = None


class VersionControl:
    """Version control system for content management."""
    
    def __init__(self, storage_path: str = "./storage/versions"):
        """Initialize version control system."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "content").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "history").mkdir(exist_ok=True)
        
        logger.info(f"Version control initialized at {self.storage_path}")
    
    def create_content(self, content: str, title: str, description: str = "", 
                      tags: Optional[List[str]] = None, source_url: Optional[str] = None, 
                      author: str = "system") -> str:
        """Create new content with initial version."""
        try:
            content_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())
            # Fallbacks for None
            title = title or ""
            description = description or ""
            tags = tags if tags is not None else []
            # Create initial version
            version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                content=content,
                version_number=1,
                timestamp=datetime.now(),
                author=author,
                changes=["Initial version"],
                metadata={
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "source_url": source_url
                }
            )
            # Save version
            self._save_version(version)
            # Create metadata
            metadata = VersionMetadata(
                content_id=content_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                total_versions=1,
                current_version=version_id,
                status="draft",
                title=title,
                description=description,
                tags=tags,
                source_url=source_url
            )
            # Save metadata
            self._save_metadata(metadata)
            logger.info(f"Created content {content_id} with version {version_id}")
            return content_id
        except Exception as e:
            logger.error(f"Error creating content: {e}")
            return ""
    
    def create_version(self, content_id: str, content: str, changes: List[str], 
                      author: str = "system") -> Optional[str]:
        """Create a new version of existing content."""
        try:
            # Get current metadata
            metadata = self._load_metadata(content_id)
            if not metadata:
                logger.error(f"Content {content_id} not found")
                return None
            # Get current version
            current_version = self._load_version(metadata.current_version)
            if not current_version:
                logger.error(f"Current version {metadata.current_version} not found")
                return None
            # Fallbacks for None
            title = metadata.title or ""
            description = metadata.description or ""
            tags = metadata.tags if metadata.tags is not None else []
            # Create new version
            version_id = str(uuid.uuid4())
            new_version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                content=content,
                version_number=current_version.version_number + 1,
                timestamp=datetime.now(),
                author=author,
                changes=changes,
                metadata=current_version.metadata.copy(),
                parent_version=current_version.version_id
            )
            # Save new version
            self._save_version(new_version)
            # Update metadata
            metadata.updated_at = datetime.now()
            metadata.total_versions += 1
            metadata.current_version = version_id
            # Save updated metadata
            self._save_metadata(metadata)
            logger.info(f"Created version {version_id} for content {content_id}")
            return version_id
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            return None
    
    def get_version(self, version_id: str) -> Optional[ContentVersion]:
        """Get a specific version by ID."""
        try:
            return self._load_version(version_id)
        except Exception as e:
            logger.error(f"Error getting version {version_id}: {e}")
            return None
    
    def get_current_version(self, content_id: str) -> Optional[ContentVersion]:
        """Get the current version of content."""
        try:
            metadata = self._load_metadata(content_id)
            if not metadata:
                return None
            
            return self._load_version(metadata.current_version)
        except Exception as e:
            logger.error(f"Error getting current version for {content_id}: {e}")
            return None
    
    def get_latest_version_id(self, content_id: str) -> Optional[str]:
        """Get the latest version ID for content."""
        try:
            metadata = self._load_metadata(content_id)
            if not metadata:
                return None
            
            return metadata.current_version
        except Exception as e:
            logger.error(f"Error getting latest version ID for {content_id}: {e}")
            return None
    
    def get_version_history(self, content_id: str) -> List[ContentVersion]:
        """Get the complete version history for content."""
        try:
            metadata = self._load_metadata(content_id)
            if not metadata:
                return []
            
            versions = []
            current_version_id = metadata.current_version
            
            # Traverse version chain
            while current_version_id:
                version = self._load_version(current_version_id)
                if version:
                    versions.append(version)
                    current_version_id = version.parent_version
                else:
                    break
            
            # Sort by version number
            versions.sort(key=lambda v: v.version_number)
            return versions
            
        except Exception as e:
            logger.error(f"Error getting version history for {content_id}: {e}")
            return []
    
    def update_status(self, version_id: str, status: str) -> bool:
        """Update the status of a version."""
        try:
            version = self._load_version(version_id)
            if not version:
                return False
            
            version.status = status
            self._save_version(version)
            
            # Update metadata if this is the current version
            metadata = self._load_metadata(version.content_id)
            if metadata and metadata.current_version == version_id:
                metadata.status = status
                metadata.updated_at = datetime.now()
                self._save_metadata(metadata)
            
            logger.info(f"Updated status of version {version_id} to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """Compare two versions and return differences."""
        try:
            version_1 = self._load_version(version_id_1)
            version_2 = self._load_version(version_id_2)
            
            if not version_1 or not version_2:
                return {"error": "One or both versions not found"}
            
            # Simple text comparison (in production, use diff libraries)
            content_1 = version_1.content
            content_2 = version_2.content
            
            # Calculate basic metrics
            chars_1 = len(content_1)
            chars_2 = len(content_2)
            words_1 = len(content_1.split())
            words_2 = len(content_2.split())
            
            return {
                "version_1": {
                    "id": version_id_1,
                    "number": version_1.version_number,
                    "timestamp": version_1.timestamp.isoformat(),
                    "author": version_1.author,
                    "characters": chars_1,
                    "words": words_1
                },
                "version_2": {
                    "id": version_id_2,
                    "number": version_2.version_number,
                    "timestamp": version_2.timestamp.isoformat(),
                    "author": version_2.author,
                    "characters": chars_2,
                    "words": words_2
                },
                "differences": {
                    "character_change": chars_2 - chars_1,
                    "word_change": words_2 - words_1,
                    "changes_1": version_1.changes,
                    "changes_2": version_2.changes
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
    
    def search_versions(self, query: str, content_id: Optional[str] = None) -> List[ContentVersion]:
        """Search through versions by content or metadata."""
        try:
            results = []
            
            if content_id:
                # Search within specific content
                versions = self.get_version_history(content_id)
                for version in versions:
                    if query.lower() in version.content.lower():
                        results.append(version)
            else:
                # Search all content
                metadata_files = list((self.storage_path / "metadata").glob("*.json"))
                
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_data = json.load(f)
                        
                        # Check metadata
                        if (query.lower() in metadata_data.get("title", "").lower() or
                            query.lower() in metadata_data.get("description", "").lower() or
                            any(query.lower() in tag.lower() for tag in metadata_data.get("tags", []))):
                            
                            # Get current version
                            current_version = self._load_version(metadata_data["current_version"])
                            if current_version:
                                results.append(current_version)
                    
                    except Exception as e:
                        logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching versions: {e}")
            return []
    
    def _save_version(self, version: ContentVersion) -> None:
        """Save version to storage."""
        version_file = self.storage_path / "content" / f"{version.version_id}.json"
        
        # Convert datetime to string for JSON serialization
        version_data = asdict(version)
        version_data["timestamp"] = version.timestamp.isoformat()
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
    
    def _load_version(self, version_id: str) -> Optional[ContentVersion]:
        """Load version from storage."""
        version_file = self.storage_path / "content" / f"{version_id}.json"
        
        if not version_file.exists():
            return None
        
        try:
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            # Convert timestamp string back to datetime
            version_data["timestamp"] = datetime.fromisoformat(version_data["timestamp"])
            
            return ContentVersion(**version_data)
            
        except Exception as e:
            logger.error(f"Error loading version {version_id}: {e}")
            return None
    
    def _save_metadata(self, metadata: VersionMetadata) -> None:
        """Save metadata to storage."""
        metadata_file = self.storage_path / "metadata" / f"{metadata.content_id}.json"
        
        # Convert datetime to string for JSON serialization
        metadata_data = asdict(metadata)
        metadata_data["created_at"] = metadata.created_at.isoformat()
        metadata_data["updated_at"] = metadata.updated_at.isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
    
    def _load_metadata(self, content_id: str) -> Optional[VersionMetadata]:
        """Load metadata from storage."""
        metadata_file = self.storage_path / "metadata" / f"{content_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
            
            # Convert timestamp strings back to datetime
            metadata_data["created_at"] = datetime.fromisoformat(metadata_data["created_at"])
            metadata_data["updated_at"] = datetime.fromisoformat(metadata_data["updated_at"])
            
            return VersionMetadata(**metadata_data)
            
        except Exception as e:
            logger.error(f"Error loading metadata for {content_id}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            content_files = list((self.storage_path / "content").glob("*.json"))
            metadata_files = list((self.storage_path / "metadata").glob("*.json"))
            
            total_content = len(metadata_files)
            total_versions = len(content_files)
            
            return {
                "total_content": total_content,
                "total_versions": total_versions,
                "storage_path": str(self.storage_path),
                "content_files": len(content_files),
                "metadata_files": len(metadata_files)
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    def list_all_content(self) -> List[Dict[str, Any]]:
        """List all content with their metadata."""
        try:
            content_list = []
            metadata_files = list((self.storage_path / "metadata").glob("*.json"))
            
            for metadata_file in metadata_files:
                try:
                    content_id = metadata_file.stem  # Remove .json extension
                    metadata = self._load_metadata(content_id)
                    
                    if metadata:
                        # Get current version content
                        current_version = self._load_version(metadata.current_version)
                        content = current_version.content if current_version else ""
                        
                        content_list.append({
                            "id": content_id,
                            "title": metadata.title,
                            "description": metadata.description,
                            "content": content,
                            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
                            "updated_at": metadata.updated_at.isoformat() if metadata.updated_at else None,
                            "author": "Unknown",  # Not stored in metadata
                            "tags": metadata.tags,
                            "total_versions": metadata.total_versions,
                            "status": metadata.status,
                            "source_url": metadata.source_url
                        })
                except Exception as e:
                    logger.warning(f"Error loading content {metadata_file.stem}: {e}")
                    continue
            
            # Sort by creation date (newest first)
            content_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return content_list
            
        except Exception as e:
            logger.error(f"Error listing all content: {e}")
            return []

    def get_all_content(self) -> Dict[str, Dict[str, Any]]:
        """Get all content metadata (legacy method for compatibility)."""
        try:
            content_dict = {}
            content_list = self.list_all_content()
            
            for content in content_list:
                content_dict[content["id"]] = {
                    "title": content["title"],
                    "description": content["description"],
                    "created_at": content["created_at"],
                    "updated_at": content["updated_at"],
                    "word_count": len(content["content"].split()),
                    "tags": content["tags"],
                    "status": content["status"]
                }
            
            return content_dict
            
        except Exception as e:
            logger.error(f"Error getting all content: {e}")
            return {} 
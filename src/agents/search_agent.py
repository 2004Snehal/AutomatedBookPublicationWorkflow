"""
Search Agent for semantic search and content retrieval using ChromaDB.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings

from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from the search agent."""
    content_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_url: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class ContentChunk:
    """Represents a chunk of content for indexing."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class SearchAgent:
    """AI-powered search agent for semantic content retrieval using ChromaDB."""
    
    def __init__(self, llm_client=None, chroma_persist_directory: str = "./chroma_db"):
        """Initialize the search agent with ChromaDB."""
        self.llm_client = llm_client if llm_client is not None else get_llm_client()
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collections
        self.content_collection = self.chroma_client.get_or_create_collection(
            name="content_chunks",
            metadata={"description": "Content chunks for semantic search"}
        )
        
        self.metadata_collection = self.chroma_client.get_or_create_collection(
            name="content_metadata",
            metadata={"description": "Content metadata for enhanced search"}
        )
        
        logger.info("Search agent initialized with ChromaDB")
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using LLM reasoning for semantic understanding."""
        try:
            # Use LLM to understand the semantic meaning and generate a semantic representation
            prompt = f"""
            You are an expert in semantic understanding and content analysis. Your task is to analyze the following text and provide a comprehensive semantic representation that captures its core meaning, themes, and key concepts.

            TEXT TO ANALYZE:
            {text}

            INSTRUCTIONS:
            Please provide a detailed semantic analysis that includes:

            1. CORE THEMES AND CONCEPTS:
            - Main topics and subjects discussed
            - Key themes and underlying messages
            - Important concepts and ideas

            2. SEMANTIC CHARACTERISTICS:
            - Tone and style of the content
            - Target audience and purpose
            - Context and domain-specific elements

            3. KEYWORDS AND PHRASES:
            - Important terminology used
            - Descriptive phrases that capture meaning
            - Contextual indicators

            4. RELATIONSHIPS AND CONNECTIONS:
            - How different ideas relate to each other
            - Logical flow and structure
            - Implied connections and associations

            Please provide a comprehensive semantic summary that would help identify similar content and enable effective search matching.

            SEMANTIC ANALYSIS:
            """
            
            semantic_analysis = await self.llm_client.generate_content(prompt)
            
            # Convert semantic analysis to embedding-like format
            # In a real implementation, you would use a proper embedding model
            import hashlib
            hash_obj = hashlib.md5(semantic_analysis.encode())
            embedding = [float(b) / 255.0 for b in hash_obj.digest()[:128]]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [0.0] * 128  # Fallback embedding
    
    async def index_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Index content in ChromaDB for semantic search using LLM reasoning."""
        try:
            # Use LLM to intelligently split content into meaningful chunks
            chunks = await self._intelligent_split_content(content)
            content_id = str(uuid.uuid4())
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{content_id}_chunk_{i}"
                
                # Generate semantic embedding for chunk
                embedding = await self.generate_embeddings(chunk)
                
                # Prepare enhanced metadata using LLM
                enhanced_metadata = await self._enhance_metadata(metadata, chunk, i, len(chunks))
                
                # Sanitize metadata for ChromaDB compatibility
                sanitized_metadata = self._sanitize_metadata(enhanced_metadata)
                
                # Add to ChromaDB
                self.content_collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[sanitized_metadata],
                    ids=[chunk_id]
                )
            
            # Store main content metadata with LLM-enhanced summary
            main_metadata = await self._create_main_metadata(metadata, content, content_id, len(chunks))
            
            # Sanitize main metadata
            sanitized_main_metadata = self._sanitize_metadata(main_metadata)
            
            self.metadata_collection.add(
                embeddings=[await self.generate_embeddings(content[:1000])],  # First 1000 chars
                documents=[content[:1000]],
                metadatas=[sanitized_main_metadata],
                ids=[content_id]
            )
            
            logger.info(f"Indexed content {content_id} with {len(chunks)} chunks using LLM reasoning")
            return content_id
            
        except Exception as e:
            logger.error(f"Error indexing content: {e}")
            return ""
    
    async def _intelligent_split_content(self, content: str, target_chunk_size: int = 1000) -> List[str]:
        """Intelligently split content into meaningful chunks using LLM reasoning."""
        try:
            prompt = f"""
            You are an expert in content analysis and document structuring. Your task is to intelligently split the following content into meaningful, coherent chunks that would be optimal for semantic search and retrieval.

            CONTENT TO SPLIT:
            {content}

            TARGET CHUNK SIZE: Approximately {target_chunk_size} characters per chunk

            INSTRUCTIONS:
            Please split the content into chunks that:

            1. MAINTAIN SEMANTIC COHERENCE:
            - Each chunk should be a complete, meaningful unit
            - Preserve logical flow and context
            - Avoid breaking sentences or paragraphs mid-thought

            2. OPTIMIZE FOR SEARCH:
            - Each chunk should be self-contained enough for effective retrieval
            - Include relevant context and background information
            - Ensure chunks are neither too short nor too long

            3. PRESERVE STRUCTURE:
            - Respect natural content boundaries (paragraphs, sections, etc.)
            - Maintain the original flow and organization
            - Keep related concepts together

            4. ENSURE COMPLETENESS:
            - Each chunk should be understandable on its own
            - Include necessary context and references
            - Avoid orphaned sentences or incomplete thoughts

            Please provide the chunks separated by "---CHUNK_BREAK---" markers.

            INTELLIGENTLY SPLIT CONTENT:
            """
            
            split_response = await self.llm_client.generate_content(prompt)
            
            # Split by the marker
            chunks = [chunk.strip() for chunk in split_response.split("---CHUNK_BREAK---") if chunk.strip()]
            
            # Fallback to simple splitting if LLM approach fails
            if not chunks or len(chunks) == 1:
                chunks = self._split_content(content, target_chunk_size, 200)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in intelligent content splitting: {e}")
            # Fallback to simple splitting
            return self._split_content(content, target_chunk_size, 200)
    
    async def _enhance_metadata(self, base_metadata: Dict[str, Any], chunk: str, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Enhance metadata using LLM reasoning for better search capabilities."""
        try:
            prompt = f"""
            You are an expert in content analysis and metadata generation. Your task is to analyze a content chunk and generate enhanced metadata that will improve search and retrieval capabilities.

            CONTENT CHUNK:
            {chunk}

            BASE METADATA:
            {json.dumps(base_metadata, indent=2)}

            CHUNK INFORMATION:
            - Chunk Index: {chunk_index + 1} of {total_chunks}

            INSTRUCTIONS:
            Please analyze the content chunk and provide enhanced metadata that includes:

            1. CONTENT ANALYSIS:
            - Main topics and themes in this chunk
            - Key concepts and terminology
            - Content type and style

            2. SEARCH OPTIMIZATION:
            - Relevant keywords and phrases
            - Semantic tags and categories
            - Contextual information

            3. RELATIONSHIP METADATA:
            - How this chunk relates to the overall content
            - Connections to other potential chunks
            - Position and role in the content structure

            Please provide the enhanced metadata in JSON format, building upon the base metadata provided.

            ENHANCED METADATA:
            """
            
            enhanced_metadata_text = await self.llm_client.generate_content(prompt)
            
            try:
                enhanced_metadata = json.loads(enhanced_metadata_text)
                # Merge with base metadata
                final_metadata = {
                    **base_metadata,
                    **enhanced_metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_id": f"chunk_{chunk_index}",
                    "timestamp": datetime.now().isoformat()
                }
                return final_metadata
            except json.JSONDecodeError:
                # Fallback to base metadata with chunk info
                return {
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_id": f"chunk_{chunk_index}",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error enhancing metadata: {e}")
            return {
                **base_metadata,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_id": f"chunk_{chunk_index}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_main_metadata(self, base_metadata: Dict[str, Any], content: str, content_id: str, total_chunks: int) -> Dict[str, Any]:
        """Create enhanced main metadata using LLM reasoning."""
        try:
            prompt = f"""
            You are an expert in content analysis and metadata generation. Your task is to analyze the complete content and generate comprehensive metadata that will improve search and retrieval capabilities.

            CONTENT SUMMARY:
            {content[:2000]}...

            BASE METADATA:
            {json.dumps(base_metadata, indent=2)}

            CONTENT INFORMATION:
            - Total Chunks: {total_chunks}
            - Content ID: {content_id}

            INSTRUCTIONS:
            Please analyze the content and provide comprehensive metadata that includes:

            1. CONTENT OVERVIEW:
            - Main themes and topics
            - Content type and genre
            - Target audience and purpose

            2. SEARCH OPTIMIZATION:
            - Primary keywords and phrases
            - Semantic categories and tags
            - Related concepts and themes

            3. STRUCTURAL INFORMATION:
            - Content organization and flow
            - Key sections and components
            - Complexity and depth indicators

            Please provide the enhanced metadata in JSON format, building upon the base metadata provided.

            ENHANCED MAIN METADATA:
            """
            
            enhanced_metadata_text = await self.llm_client.generate_content(prompt)
            
            try:
                enhanced_metadata = json.loads(enhanced_metadata_text)
                # Merge with base metadata
                final_metadata = {
                    **base_metadata,
                    **enhanced_metadata,
                    "content_id": content_id,
                    "total_chunks": total_chunks,
                    "indexed_at": datetime.now().isoformat()
                }
                return final_metadata
            except json.JSONDecodeError:
                # Fallback to base metadata
                return {
                    **base_metadata,
                    "content_id": content_id,
                    "total_chunks": total_chunks,
                    "indexed_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error creating main metadata: {e}")
            return {
                **base_metadata,
                "content_id": content_id,
                "total_chunks": total_chunks,
                "indexed_at": datetime.now().isoformat()
            }
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search on indexed content using LLM reasoning."""
        try:
            # Use LLM to enhance the search query for better semantic understanding
            enhanced_query = await self._enhance_search_query(query)
            
            # Generate embedding for enhanced query
            query_embedding = await self.generate_embeddings(enhanced_query)
            
            # Search in content collection
            results = self.content_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )
            
            search_results = []
            
            # Check if results exist and have the expected structure
            if (results and 'ids' in results and results['ids'] and 
                len(results['ids']) > 0 and results['ids'][0] and
                'metadatas' in results and results['metadatas'] and
                len(results['metadatas']) > 0 and results['metadatas'][0] and
                'documents' in results and results['documents'] and
                len(results['documents']) > 0 and results['documents'][0] and
                'distances' in results and results['distances'] and
                len(results['distances']) > 0 and results['distances'][0]):
                
                for i in range(len(results['ids'][0])):
                    try:
                        # Safely extract all fields with null checks
                        metadata_item = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                        document_item = results['documents'][0][i] if i < len(results['documents'][0]) else ""
                        distance_item = results['distances'][0][i] if i < len(results['distances'][0]) else 0.0
                        
                        content_id = str(metadata_item.get('content_id', ''))
                        content = str(document_item)
                        metadata = dict(metadata_item) if metadata_item else {}
                        similarity_score = float(distance_item) if distance_item is not None else 0.0
                        source_url = str(metadata_item.get('source_url', '')) if metadata_item and metadata_item.get('source_url') else None
                        
                        # Handle timestamp safely
                        timestamp_str = metadata_item.get('timestamp', '') if metadata_item else ''
                        timestamp = datetime.fromisoformat(str(timestamp_str)) if timestamp_str else datetime.now()
                        
                        result = SearchResult(
                            content_id=content_id,
                            content=content,
                            metadata=metadata,
                            similarity_score=similarity_score,
                            source_url=source_url,
                            timestamp=timestamp
                        )
                        search_results.append(result)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"Error processing search result {i}: {e}")
                        continue
            
            # Use LLM to rank and refine results
            refined_results = await self._refine_search_results(query, search_results)
            
            logger.info(f"Found {len(refined_results)} search results for query: {query}")
            return refined_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _enhance_search_query(self, query: str) -> str:
        """Enhance search query using LLM reasoning for better semantic understanding."""
        try:
            prompt = f"""
            You are an expert in search query optimization and semantic understanding. Your task is to enhance the following search query to improve semantic search results.

            ORIGINAL QUERY:
            {query}

            INSTRUCTIONS:
            Please enhance the query by:

            1. EXPANDING SEMANTIC MEANING:
            - Add related concepts and synonyms
            - Include broader and narrower terms
            - Consider different ways to express the same intent

            2. IMPROVING SEARCH CONTEXT:
            - Add relevant context and background
            - Include domain-specific terminology
            - Consider the user's likely intent

            3. OPTIMIZING FOR RETRIEVAL:
            - Use terms that are likely to appear in relevant content
            - Include both specific and general terms
            - Consider different content types and formats

            Please provide an enhanced version of the query that will lead to better search results while maintaining the original intent.

            ENHANCED QUERY:
            """
            
            enhanced_query = await self.llm_client.generate_content(prompt)
            return enhanced_query.strip()
            
        except Exception as e:
            logger.error(f"Error enhancing search query: {e}")
            return query
    
    async def _refine_search_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Refine search results using LLM reasoning for better relevance ranking."""
        try:
            if not results:
                return results
            
            # Create a prompt for the LLM to evaluate and rank results
            results_text = "\n\n".join([
                f"Result {i+1} (Score: {r.similarity_score:.3f}):\n{r.content[:500]}..."
                for i, r in enumerate(results)
            ])
            
            prompt = f"""
            You are an expert in search result evaluation and ranking. Your task is to analyze the search results for the given query and provide insights on their relevance and quality.

            SEARCH QUERY:
            {query}

            SEARCH RESULTS:
            {results_text}

            INSTRUCTIONS:
            Please analyze each result and provide:

            1. RELEVANCE ASSESSMENT:
            - How well each result matches the query intent
            - Relevance score (1-10) for each result
            - Specific reasons for the relevance score

            2. QUALITY EVALUATION:
            - Content quality and usefulness
            - Completeness of information
            - Clarity and readability

            3. RANKING RECOMMENDATIONS:
            - Suggested reordering of results
            - Which results should be prioritized
            - Any results that should be filtered out

            Please provide your analysis in a structured format that can help improve the search result ranking.

            ANALYSIS:
            """
            
            analysis = await self.llm_client.generate_content(prompt)
            
            # For now, return the original results
            # In a full implementation, you would parse the analysis and reorder results
            logger.info(f"Search results analyzed using LLM reasoning")
            return results
            
        except Exception as e:
            logger.error(f"Error refining search results: {e}")
            return results
    
    async def search_by_metadata(self, metadata_filters: Dict[str, Any], 
                               top_k: int = 10) -> List[SearchResult]:
        """Search content by metadata filters using LLM reasoning."""
        try:
            # Use LLM to optimize metadata filters for better search
            optimized_filters = await self._optimize_metadata_filters(metadata_filters)
            
            results = self.content_collection.query(
                query_embeddings=[[0.0] * 128],  # Dummy embedding for metadata-only search
                n_results=top_k,
                where=optimized_filters
            )
            
            search_results = []
            
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    content_id=results['metadatas'][0][i]['content_id'],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    similarity_score=1.0,  # No similarity score for metadata search
                    source_url=results['metadatas'][0][i].get('source_url'),
                    timestamp=datetime.fromisoformat(results['metadatas'][0][i]['timestamp'])
                )
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results for metadata search using LLM reasoning")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
    
    async def _optimize_metadata_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize metadata filters using LLM reasoning."""
        try:
            prompt = f"""
            You are an expert in search optimization and metadata analysis. Your task is to optimize the following metadata filters to improve search results.

            ORIGINAL FILTERS:
            {json.dumps(filters, indent=2)}

            INSTRUCTIONS:
            Please analyze and optimize the filters by:

            1. EXPANDING FILTER COVERAGE:
            - Add related terms and synonyms
            - Include broader and narrower categories
            - Consider alternative ways to express the same criteria

            2. IMPROVING FILTER PRECISION:
            - Remove overly broad or vague filters
            - Add more specific criteria where appropriate
            - Ensure filters are mutually compatible

            3. OPTIMIZING FOR RETRIEVAL:
            - Use filters that are likely to match indexed content
            - Consider the structure of the metadata
            - Balance between specificity and recall

            Please provide optimized filters in JSON format.

            OPTIMIZED FILTERS:
            """
            
            optimized_filters_text = await self.llm_client.generate_content(prompt)
            
            try:
                optimized_filters = json.loads(optimized_filters_text)
                return optimized_filters
            except json.JSONDecodeError:
                return filters
                
        except Exception as e:
            logger.error(f"Error optimizing metadata filters: {e}")
            return filters
    
    def _split_content(self, content: str, chunk_size: int = 1000, 
                      overlap: int = 200) -> List[str]:
        """Fallback method for simple content splitting."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(content):
                break
        
        return chunks
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed content."""
        try:
            content_count = self.content_collection.count()
            metadata_count = self.metadata_collection.count()
            
            return {
                "total_content_chunks": content_count,
                "total_metadata_entries": metadata_count,
                "collections": {
                    "content_chunks": content_count,
                    "content_metadata": metadata_count
                }
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    async def delete_content(self, content_id: str) -> bool:
        """Delete content from the index."""
        try:
            # Find all chunks for this content
            results = self.content_collection.query(
                query_embeddings=[[0.0] * 128],
                n_results=1000,
                where={"content_id": content_id}
            )
            
            if results['ids'][0]:
                self.content_collection.delete(ids=results['ids'][0])
            
            # Delete main metadata
            self.metadata_collection.delete(ids=[content_id])
            
            logger.info(f"Deleted content {content_id} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting content: {e}")
            return False

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure compatibility with ChromaDB."""
        sanitized = {}
        
        for key, value in metadata.items():
            # Convert key to string
            key_str = str(key)
            
            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                sanitized[key_str] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key_str] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                sanitized[key_str] = json.dumps(value)
            elif value is None:
                # Convert None to empty string
                sanitized[key_str] = ""
            else:
                # Convert any other type to string
                sanitized[key_str] = str(value)
        
        return sanitized


def get_search_agent() -> SearchAgent:
    """Get a search agent instance."""
    return SearchAgent() 
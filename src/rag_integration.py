"""
RAG Integration - Knowledge base retrieval for agents
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json


class SimpleRAG:
    """
    Simple RAG (Retrieval Augmented Generation) system.
    Uses basic text search - can be upgraded to vector embeddings later.
    """
    
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """
        Initialize RAG system.
        
        Args:
            knowledge_base_dir: Directory containing knowledge files
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self._ensure_kb_dir()
        self.documents = []
        self.load_knowledge_base()
    
    def _ensure_kb_dir(self):
        """Create knowledge base directory if it doesn't exist"""
        try:
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create README
            readme_path = self.knowledge_base_dir / "README.md"
            if not readme_path.exists():
                readme_path.write_text("""# Knowledge Base

Add text files (.txt, .md) or JSON files here for RAG retrieval.

## Supported Formats

- **Text files (.txt, .md)**: Plain text documents
- **JSON files (.json)**: Structured documents with metadata

## JSON Format

```json
{
  "title": "Document Title",
  "content": "Document content...",
  "metadata": {
    "category": "Category",
    "tags": ["tag1", "tag2"]
  }
}
```
""")
        except Exception as e:
            print(f"Warning: Could not create knowledge base directory: {e}")
    
    def load_knowledge_base(self):
        """Load all documents from knowledge base directory"""
        self.documents = []
        
        try:
            # Load text files
            for file_path in self.knowledge_base_dir.glob("*.txt"):
                try:
                    content = file_path.read_text(encoding='utf-8')
                    self.documents.append({
                        "filename": file_path.name,
                        "title": file_path.stem,
                        "content": content,
                        "type": "text",
                        "metadata": {}
                    })
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
            
            # Load markdown files
            for file_path in self.knowledge_base_dir.glob("*.md"):
                if file_path.name == "README.md":
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8')
                    self.documents.append({
                        "filename": file_path.name,
                        "title": file_path.stem,
                        "content": content,
                        "type": "markdown",
                        "metadata": {}
                    })
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
            
            # Load JSON files
            for file_path in self.knowledge_base_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    self.documents.append({
                        "filename": file_path.name,
                        "title": data.get("title", file_path.stem),
                        "content": data.get("content", ""),
                        "type": "json",
                        "metadata": data.get("metadata", {})
                    })
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
        
        print(f"Loaded {len(self.documents)} documents into knowledge base")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant documents.
        Uses simple keyword matching - can be upgraded to embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            return []
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        results = []
        
        for doc in self.documents:
            # Calculate relevance score (simple term matching)
            content_lower = doc["content"].lower()
            title_lower = doc["title"].lower()
            
            # Count matching terms
            content_matches = sum(1 for term in query_terms if term in content_lower)
            title_matches = sum(1 for term in query_terms if term in title_lower)
            
            # Weight title matches higher
            score = (content_matches * 1.0) + (title_matches * 3.0)
            
            if score > 0:
                results.append({
                    **doc,
                    "relevance_score": score
                })
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
    
    def get_context_for_query(self, query: str, top_k: int = 3, max_chars: int = 2000) -> str:
        """
        Get formatted context for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            max_chars: Maximum characters to return
        
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "No relevant knowledge base documents found."
        
        context_parts = ["Retrieved knowledge base information:\n"]
        total_chars = 0
        
        for idx, doc in enumerate(results, 1):
            section = f"\n[Source {idx}: {doc['title']}]\n{doc['content']}\n"
            
            if total_chars + len(section) > max_chars and idx > 1:
                break
            
            context_parts.append(section)
            total_chars += len(section)
        
        return "".join(context_parts)
    
    def add_document(
        self, 
        title: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            title: Document title
            content: Document content
            metadata: Optional metadata
            filename: Optional custom filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not filename:
                # Generate safe filename
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))
                safe_title = safe_title.replace(' ', '_')[:50]
                filename = f"{safe_title}.json"
            
            file_path = self.knowledge_base_dir / filename
            
            doc_data = {
                "title": title,
                "content": content,
                "metadata": metadata or {}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            # Reload knowledge base
            self.load_knowledge_base()
            
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a document from knowledge base.
        
        Args:
            filename: Document filename
        
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.knowledge_base_dir / filename
            
            if not file_path.exists():
                return False
            
            file_path.unlink()
            
            # Reload knowledge base
            self.load_knowledge_base()
            
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def list_documents(self) -> List[Dict[str, str]]:
        """
        List all documents in knowledge base.
        
        Returns:
            List of document summaries
        """
        return [
            {
                "filename": doc["filename"],
                "title": doc["title"],
                "type": doc["type"],
                "content_length": len(doc["content"]),
                "has_metadata": len(doc.get("metadata", {})) > 0
            }
            for doc in self.documents
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_documents": len(self.documents),
            "total_characters": sum(len(doc["content"]) for doc in self.documents),
            "document_types": {
                "text": sum(1 for doc in self.documents if doc["type"] == "text"),
                "markdown": sum(1 for doc in self.documents if doc["type"] == "markdown"),
                "json": sum(1 for doc in self.documents if doc["type"] == "json")
            }
        }


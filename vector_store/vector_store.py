"""
Vector Store Manager for AgentForge
Provides semantic search capabilities using ChromaDB and sentence-transformers
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStoreManager:
    """
    Manages vector embeddings and semantic search using ChromaDB.
    """
    
    def __init__(
        self,
        persist_directory: str = "vector_store",
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to use
            embedding_model: Sentence-transformers model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        print(f"Loading {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Model ready")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' loaded")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "AgentForge knowledge base"}
            )
            print(f"Collection '{collection_name}' created")
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Unique identifier for the document
            content: Document content
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare metadata
            meta = metadata or {}
            meta["content_length"] = len(content)
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta]
            )
            
            return True
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")
            return False
    
    def add_documents_batch(
        self,
        doc_ids: List[str],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Add multiple documents in batch.
        
        Args:
            doc_ids: List of document IDs
            contents: List of document contents
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # Prepare metadata
            if metadatas is None:
                metadatas = [{"content_length": len(c)} for c in contents]
            else:
                for i, meta in enumerate(metadatas):
                    meta["content_length"] = len(contents[i])
            
            # Add to collection
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            print(f"Error adding documents batch: {e}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the vector store.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with content, metadata, and distance
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
    
    def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            content: New content
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete old version
            self.delete_document(doc_id)
            
            # Add new version
            return self.add_document(doc_id, content, metadata)
        except Exception as e:
            print(f"Error updating document {doc_id}: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result and result['ids'] and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            return None
        except Exception as e:
            print(f"Error getting document {doc_id}: {e}")
            return None
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the collection.
        
        Returns:
            List of all documents
        """
        try:
            result = self.collection.get()
            
            documents = []
            if result and result['ids']:
                for i in range(len(result['ids'])):
                    documents.append({
                        'id': result['ids'][i],
                        'content': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {}
                    })
            
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats
        """
        try:
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_documents': 0,
                'collection_name': self.collection_name,
                'embedding_model': 'unknown'
            }
    
    def reset_collection(self) -> bool:
        """
        Delete all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AgentForge knowledge base"}
            )
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False

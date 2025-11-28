import os
import sqlite3
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleRetriever:
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks = []
        self.chunk_ids = []
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = None
        self._load_documents()
        
    def _load_documents(self):
        """Load documents and create chunks"""
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple chunking by paragraphs
                paragraphs = content.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        chunk_id = f"{filename}::chunk{i}"
                        self.chunks.append(para.strip())
                        self.chunk_ids.append(chunk_id)
        
        # Create TF-IDF vectors
        if self.chunks:
            self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, any]]:
        """Retrieve top-k chunks based on TF-IDF similarity"""
        if not self.chunks:
            return []
            
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return chunks with similarity > 0
                results.append({
                    'chunk_id': self.chunk_ids[idx],
                    'content': self.chunks[idx],
                    'score': float(similarities[idx])
                })
        
        return results
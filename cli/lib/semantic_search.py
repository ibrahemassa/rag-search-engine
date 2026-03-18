import re
from sentence_transformers import SentenceTransformer
from numpy.typing import ArrayLike
import numpy as np
import os

from sentence_transformers.util import semantic_search

from .search_utils import CACHE_DIR, CHUNK_REGEX, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNKS_SIZE, DEFAULT_MAX_CHUNK_SIZE, MOVIE_EMBEDDINGS_PATH, RESULTS_LIMIT, load_movies

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict] = []
        self.document_map = {}

        self.embeddings_path = MOVIE_EMBEDDINGS_PATH

    def generate_embedding(self, text: str) -> ArrayLike:
        text = str(text)
        if not text or not text.strip():
            raise ValueError("Empty text can't be embedded!")

        embeddeings = self.model.encode([text])
        return embeddeings[0]

    def build_embeddings(self, documents: list[dict]) -> ArrayLike:
        self.documents = documents
        texts = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_text = f"{doc['title']}: {doc['description']}"
            texts.append(doc_text)

        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> ArrayLike:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit = RESULTS_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, self.documents[i]))

        return [{
            "score": doc[0],
            "title": doc[1]["title"],
            "description": doc[1]["description"]
        } for doc in sorted(similarities, reverse=True, key=lambda x: x[0])[:limit]]


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
     
def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search_command(query: str, limit=RESULTS_LIMIT):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    results = ss.search(query, limit)

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(res["description"][:100], "...")

def fixed_size_chunk(text: str, size: int = DEFAULT_CHUNKS_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    count_words = len(words)
    chunks = []

    i = 0
    while i < count_words:
        chunk = words[i:i + size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i = i + size - overlap

    return chunks

def chunk_text(text: str, size: int = DEFAULT_CHUNKS_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunk(text, size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

# def semantic_chunk(text: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
#     sentences = re.split(CHUNK_REGEX, text)
#     count_sentences = len(sentences)
#     chunks = []
#     i = 0
#     while i < count_sentences:
#         chunk = sentences[i:i + max_size]
#         if chunks and len(chunk) <= overlap:
#             break
#         chunks.append(" ".join(chunk))
#         i = i + max_size - overlap
#
#     return chunks
#
#
# def semantic_chunk_command(text: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
#     chunks = semantic_chunk(text, max_size, overlap)
#     print(f"Semantically chunking {len(text)} characters")
#     for i, chunk in enumerate(chunks, 1):
#         print(f"{i}. {chunk}")
#

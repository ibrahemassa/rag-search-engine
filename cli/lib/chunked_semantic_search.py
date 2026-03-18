from collections import defaultdict
import json
import re


from .search_utils import CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, CHUNK_REGEX, DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_CHUNK_SIZE, DOCUMENT_PREVIEW_LENGTH, SCORE_PRECISION, load_movies
from .semantic_search import SemanticSearch, chunk_text, cosine_similarity 
import numpy as np
import os


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray = None
        self.chunk_metadata: list[dict] = []

        self.chunk_embeddings_path = CHUNK_EMBEDDINGS_PATH
        self.chunk_metadata_path = CHUNK_METADATA_PATH
 
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        chunks: list[str] = []
        metadata: list[dict] = []
        for movie_idx, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            desc = doc["description"].strip()
            if desc:
                desc_chunks = semantic_chunk(desc, 4, 1)
                for chunk_idx, chunk in enumerate(desc_chunks):
                    chunks.append(chunk)
                    meta = {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(desc_chunks)
                    }
                    metadata.append(meta)
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        chunks_scores: list[dict] = []
        scores_dict = defaultdict(float)
        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk)
            movie_idx = self.chunk_metadata[i]["movie_idx"]
            data = {
                "chunk_idx": i,
                "movie_idx": movie_idx,
                "score": score
            }

            chunks_scores.append(data)
            scores_dict[movie_idx] = max(scores_dict[movie_idx], score)

        sorted_scores = [(idx, score) for idx, score in sorted(scores_dict.items(), reverse=True, key=lambda x: x[1])][:limit]
        result = []
        for s in sorted_scores:
            doc = self.documents[s[0]]
            result.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                "score": round(s[1], SCORE_PRECISION),
            })

        return result

def semantic_chunk(text: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences_row = re.split(CHUNK_REGEX, text)
    sentences = []
    for sentence in sentences_row:
        sentence = sentence.strip()
        if sentence: sentences.append(sentence)

    count_sentences = len(sentences)
    if count_sentences == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0
    while i < count_sentences:
        chunk = sentences[i:i + max_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i = i + max_size - overlap

    return chunks


def semantic_chunk_command(text: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    chunks = semantic_chunk(text, max_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks():
    documents = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int = 10):
    documents = load_movies()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(documents)

    results = css.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


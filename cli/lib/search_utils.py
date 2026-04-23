from typing import Any
import json
import os


RESULTS_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100
DEFAULT_ALPHA = 0.5
DEFAULT_K = 60
SEARCH_MULTIPLIER = 5
DEFAULT_EVALUATION_K = 5

DEFAULT_CHUNKS_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_MAX_CHUNK_SIZE = 4
CHUNK_REGEX = r"(?<=[.!?])\s+"

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

ENHANCE_METHODS = ['spell', 'rewrite', 'expand']
RERANK_METHODS = ['individual', 'batch', 'cross_encoder']



def load_movies() -> list[dict]:
    with open(DATA_PATH, 'r') as data:
        movies = json.load(data)

    return movies['movies']

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, 'r') as f:
        words = f.read().splitlines()
    return words

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


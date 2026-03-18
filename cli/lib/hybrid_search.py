import os

from lib.query_enhancement import enhance_query
from lib.search_utils import DEFAULT_ALPHA, ENHANCE_METHODS, RESULTS_LIMIT, DOCUMENT_PREVIEW_LENGTH, load_movies, DEFAULT_K

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def __bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def __get_results(self, query: str, limit: int = RESULTS_LIMIT):
        bm25_results = self.__bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        return bm25_results, semantic_results

    def weighted_search(self, query: str, alpha=DEFAULT_ALPHA, limit=RESULTS_LIMIT):
        bm25_results, semantic_results = self.__get_results(query, limit)
        bm25_normalized = normalize([res["score"] for res in bm25_results])
        semantic_normalized = normalize([res["score"] for res in semantic_results])

        combined = {}

        for i, res in enumerate(bm25_results):
            res["score"] = bm25_normalized[i]
            if res["id"] not in combined:
                combined[res["id"]] = {
                    "document": res["document"],
                    "title": res["title"],
                    "bm25": res["score"],
                    "semantic": 0
                }

        for i, res in enumerate(semantic_results):
            res["score"] = semantic_normalized[i]
            if res["id"] not in combined:
                combined[res["id"]] = {
                    "document": res["document"],
                    "title": res["title"],
                    "bm25": 0,
                    "semantic": res["score"]
                }
            else:
                combined[res["id"]]["semantic"] = res["score"]

        for i in combined:
            combined[i]["hybrid_score"] = hybrid_score(combined[i]["bm25"], combined[i]["semantic"], alpha)

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results, semantic_results = self.__get_results(query, limit)
        combined = {}

        for i, res in enumerate(bm25_results, 1):
            if res["id"] not in combined:
                combined[res["id"]] = {
                    "document": res["document"],
                    "title": res["title"],
                    "bm25": i,
                    "semantic": -1,
                    "rrf": 0
                }

        for i, res in enumerate(semantic_results, 1):
            # res["score"] = semantic_normalized[i]
            if res["id"] not in combined:
                combined[res["id"]] = {
                    "document": res["document"],
                    "title": res["title"],
                    "bm25": -1,
                    "semantic": i,
                    "rrf": 0
                }
            else:
                combined[res["id"]]["semantic"] = i

        for i in combined:
            combined[i]["rrf"] += rrf_score(combined[i]["bm25"], k)
            combined[i]["rrf"] += rrf_score(combined[i]["semantic"], k)

        return sorted(combined.values(), key=lambda x: x["rrf"], reverse=True)[:limit]

def normalize(values: list[float]) -> list[float]:
    values = list(map(float, values))
    min_, max_ = min(values), max(values)
    if min_ == max_:
        return [1.00] * len(values)
    denom = max_ - min_
    normalized = [(i - min_) / denom for i in values]

    return normalized

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=DEFAULT_K):
    if rank == -1:
        return 0
    return 1 / (k + rank)

def weighted_search_command(query, alpha=DEFAULT_ALPHA, limit=RESULTS_LIMIT):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        print(f"Hybrid Score: {res['hybrid_score']:.4f}")
        print(f"BM25: {res['bm25']:.4f}, Semantic: {res['semantic']:.4f}")
        print(res["document"][:DOCUMENT_PREVIEW_LENGTH])

def rrf_search_command(query, k=DEFAULT_K, limit=10, enhance=""):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    if enhance in ENHANCE_METHODS:
        enhanced_query = enhance_query(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query
    results = hybrid_search.rrf_search(query, k, limit)

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        print(f"RRF Score: {res['rrf']:.4f}")
        print(f"BM25 Rank: {res['bm25']:.4f}, Semantic Rank: {res['semantic']:.4f}")
        print(res["document"][:DOCUMENT_PREVIEW_LENGTH])


import pickle
import heapq
from .search_utils import BM25_B, format_search_result, load_movies, BM25_K1, RESULTS_LIMIT
from .text_processing import tokenize
from collections import defaultdict, Counter
from .search_utils import CACHE_DIR
import math
import os


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = defaultdict(int)
        
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        term = term.lower()
        indicies = self.index.get(term, set())
        return sorted(list(indicies))

    def __get_avg_doc_length(self) -> float:
        total = 0 
        for ln in self.doc_lengths:
            total += self.doc_lengths[ln]

        num_docs = len(self.doc_lengths)
        return total / num_docs if num_docs > 0 else 0

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        return self.term_frequencies[doc_id].get(token, 0) 

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        frequency = len(self.index[token])   
        return math.log((len(self.docmap) + 1) / (frequency + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        tf_idf = tf * idf
        return tf_idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        token = tokens[0]

        frequency = len(self.index[token])   
        num_docs = len(self.docmap)
        return math.log((num_docs - frequency + 0.5) / (frequency + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        ln = self.doc_lengths[doc_id]
        avg = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (ln / avg) if avg > 0 else 1

        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit=RESULTS_LIMIT) -> list[dict]:
        tokens = tokenize(query)
        min_heap = []

        for doc_id in self.docmap:
            score = 0
            for token in tokens:
                score += self.bm25(doc_id, token)

            heapq.heappush(min_heap, (score, doc_id))

            if len(min_heap) > limit:
                heapq.heappop(min_heap)

        results = []
        for score, doc_id in sorted(min_heap, reverse=True):
            movie = self.docmap[doc_id]
            result = format_search_result(doc_id, movie["title"], movie["description"], score)
            results.append(result)
            # result.append(f"({movie['id']}) {movie["title"]} - score: {score: .2f}")

        return results

       
    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            doc_description = f'{movie["title"]} {movie["description"]}'
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except:
            print("File/s not found!")

def build_command() -> None:
    ii = InvertedIndex()
    ii.build()
    ii.save()

def term_frequencies_command(doc_id: int, term: str) -> int:
    ii = InvertedIndex()
    ii.load()
    frequency = ii.get_tf(doc_id, term)
    return frequency


def idf_command(term: str) -> float:
    ii = InvertedIndex()
    ii.load()
    return ii.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    ii = InvertedIndex()
    ii.load()
    return ii.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    ii = InvertedIndex()
    ii.load()
    return ii.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    ii = InvertedIndex()
    ii.load()
    bm25_tf = ii.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf

def bm25_search_command(query: str, limit=RESULTS_LIMIT) -> list[str]:
    ii = InvertedIndex()
    ii.load()
    results = ii.bm25_search(query, limit)
    return results


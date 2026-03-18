from .search_utils import RESULTS_LIMIT, load_movies
from.inverted_index import InvertedIndex
from .text_processing import text_preprocessing, tokenize, tokens_matching

def search_command(keyword: str, limit = RESULTS_LIMIT) -> list[dict]:
    found = []
    # movies = load_movies()
    index = InvertedIndex()
    index.load()
    query_tokens = tokenize(keyword)
    for token in query_tokens:
        result = index.get_documents(token)
        # found += result[:limit]
        for doc in result:
            if doc not in found:
                found.append(doc)

            if len(found) >= limit:
                break
        
    # for m in movies:
    #     title_tokens = tokenize(m['title'])
    #     if tokens_matching(query_tokens, title_tokens):
    #         found.append(m)
    #

    return list(map(index.docmap.get, found))
    # return [index.docmap[i] for i in found]
    # return found



import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import (
    RESULTS_LIMIT ,
    DEFAULT_K,
    SEARCH_MULTIPLIER, load_movies,
)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def generate_answer(search_results, query, limit=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {context}

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    return (response.text or "").strip()


def augmented_generation_helper(query: str, limit=RESULTS_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=DEFAULT_K, limit=limit * SEARCH_MULTIPLIER
    )

    return {
        "query": query,
        "search_results": search_results[:limit],
    }

    # answer = generate_answer(search_results, query, limit)
    #
    # return {
    #     "query": query,
    #     "search_results": search_results[:limit],
    #     "answer": answer,
    # }
    #
def rag(query):
    results =  augmented_generation_helper(query)
    if "search_results" not in results or not results["search_results"]:
        return {**results, "answer": "No answer"}
    answer = generate_answer(results["search_results"], query)

    return {**results, "answer": answer}

def summarize(query, limit=RESULTS_LIMIT):
    results = augmented_generation_helper(query, limit)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {results['search_results']}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return {**results, "answer": answer}

def citations(query, limit=RESULTS_LIMIT):
    results = augmented_generation_helper(query, limit)
    prompt = prompt = f"""Answer the query below and give information based on the provided documents.

    The answer should be tailored to users of Hoopla, a movie streaming service.
    If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

    Query: {query}

    Documents:
    {results["search_results"]}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources in the format [1], [2], etc. when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the provided documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return {**results, "answer": answer}

def answer_question(question: str, limit=RESULTS_LIMIT):
    results = augmented_generation_helper(question, limit)
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

    Question: {question}

    Documents:
    {results["search_results"]}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return {**results, "answer": answer}


def rag_command(query):
    return rag(query)

def summarize_command(query, limit=RESULTS_LIMIT):
    return summarize(query, limit)

def citations_command(query, limit=RESULTS_LIMIT):
    return citations(query, limit)

def question_command(question, limit=RESULTS_LIMIT):
    return answer_question(question, limit)

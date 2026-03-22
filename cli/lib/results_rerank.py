import json
import os
import time 
from typing import Optional

from dotenv import load_dotenv
from google import genai
from sentence_transformers.util import pairwise_angle_sim

from lib.search_utils import RESULTS_LIMIT

from sentence_transformers import CrossEncoder, cross_encoder


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
MODEL = "gemma-3-27b-it"

def individual_rerank(query: str, docs: list[dict], original_limit: int) -> list[dict]:
    for doc in docs:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=MODEL, contents=prompt)
        try:
            rank = int((response.text or "").strip().strip('"'))
        except Exception as e:
            print("Rerank failed:", e)
            rank = 0

        doc["rerank_score"] = rank if rank else 0
        time.sleep(3)

    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:original_limit]

def batch_rerank(query: str, docs: list[dict], original_limit: int) -> list[dict]:
    doc_map = {doc["id"]: doc for doc in docs}
    doc_list = []
    for doc in docs:
        doc_id = doc["id"]
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.models.generate_content(model=MODEL, contents=prompt)
    response = response.text or ""
    response = json.loads(response)

    try:
        return [{**doc_map[id], "rerank_rank": i} for i, id in enumerate(response, 1)][:original_limit]
    except Exception as e:
        print("Rerank failed:", e)
        return docs

def cross_encoder_rerank(query: str, docs: list[dict], original_limit: int) -> list[dict]:
    pairs = [[query, f"{doc.get('title', '')} - {doc.get('document', '')}"] for doc in docs]

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    scores_dict = [{**docs[i], "cross_encoder_score": float(score)} for i, score in enumerate(scores)][:original_limit]
    return sorted(scores_dict, key=lambda x: x["cross_encoder_score"], reverse=True)[:original_limit]

def rerank_results(query: str, docs: list[dict], method: Optional[str]="batch", original_limit: int=RESULTS_LIMIT) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(query, docs, original_limit)
        case "batch":
            return batch_rerank(query, docs, original_limit)
        case "cross_encoder":
            return cross_encoder_rerank(query, docs, original_limit)
        case _:
            return docs


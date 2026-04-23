import json
from lib.hybrid_search import rrf_search_command
from lib.search_utils import DEFAULT_EVALUATION_K, GOLDEN_DATASET_PATH


def  load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        data = json.load(f)

    return data['test_cases']

def calculate_precision(actual, expected, k) -> float:
    num_relevent = 0 
    for doc in actual:
        if doc in expected:
            num_relevent += 1
    return num_relevent / k

def calculate_recall(actual, expected, k) -> float:
    num_relevent = 0 
    for doc in actual:
        if doc in expected:
            num_relevent += 1
    return num_relevent / len(expected)


def precision_command(k=DEFAULT_EVALUATION_K):
    precisions = []
    recalls = []
    test_cases = load_golden_dataset()
    for case in test_cases[:3]:
        results = rrf_search_command(query=case["query"], limit=k)
        results = list(map(lambda x: x["title"], results))
        precision = calculate_precision(results, case["relevant_docs"], k)
        precisions.append(precision)

        recall = calculate_recall(results, case["relevant_docs"], k)
        recalls.append(recall)

        f1 = 2 * (precision * recall) / (precision + recall)


        print(f"-Query: {case['query']}")
        print(f" -Precision@{k}:{precision: .4f}")
        print(f" -Recall@{k}:{recall: .4f}")
        print(f" -F1 Score:{f1: .4f}")
        print(f" -Retrieved: {(', ').join(results)}")
        print(f" -Relevant: {(', ').join(case['relevant_docs'])}")

    return precisions


# - Query: dangerous bear wilderness survival
#   - Precision@6: 1.0000
#   - Retrieved: The Edge, Man in the Wilderness, Claws, Unnatural, Into the Grizzly Maze, Alaska
#   - Relevant: Unnatural, Alaska, The Edge, Into the Grizzly Maze, Claws, Man in the Wilderness, The Revenant




    # {
    #   "query": "cute british bear marmalade",
    #   "relevant_docs": ["Paddington"]
    # },

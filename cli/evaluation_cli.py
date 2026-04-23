import argparse

from lib.search_utils import DEFAULT_EVALUATION_K
from lib.evaluation import precision_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_EVALUATION_K,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    precision = precision_command(k=limit)

if __name__ == "__main__":
    main()

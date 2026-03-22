import argparse

from lib.search_utils import DEFAULT_ALPHA, DEFAULT_K, ENHANCE_METHODS, RESULTS_LIMIT, RERANK_METHODS
from lib.hybrid_search import normalize, weighted_search_command, rrf_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parser.add_parser("normalize", help="Normalize a list of numbers")
    normalize_parser.add_argument("values", nargs="+", type=float, help="List of numberes to normalize")

    weighted_search_parser = sub_parser.add_parser("weighted-search", help="Search for a query using weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", nargs="?", type=float, default=DEFAULT_ALPHA, help="Alpha value")
    weighted_search_parser.add_argument("--limit", nargs="?", type=int, default=RESULTS_LIMIT, help="Results limit")

    rrf_search_parser = sub_parser.add_parser("rrf-search", help="Search for a query using rrf search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", nargs="?", type=int, default=DEFAULT_K, help="K value")
    rrf_search_parser.add_argument("--limit", nargs="?", type=int, default=RESULTS_LIMIT, help="Results limit")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=ENHANCE_METHODS,
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=RERANK_METHODS,
        help="Results re-ranking method",
    )


    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.values)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(query=args.query, k=args.k, limit=args.limit, enhance=args.enhance, rerank=args.rerank_method)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

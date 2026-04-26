import argparse

from lib.augmented_generation import citations_command, question_command, rag_command, summarize_command
from lib.search_utils import RESULTS_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Use LLM to summarize hybrid search results")
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", nargs="?", type=int, default=RESULTS_LIMIT, help="Results limit")

    citations_parser = subparsers.add_parser("citations", help="Use LLM to summarize hybrid search results")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", nargs="?", type=int, default=RESULTS_LIMIT, help="Results limit")

    question_parser = subparsers.add_parser("question", help="Use LLM to summarize hybrid search results")
    question_parser.add_argument("question", type=str, help="Question")
    question_parser.add_argument("--limit", nargs="?", type=int, default=RESULTS_LIMIT, help="Results limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Sumamry:")
            print(result["answer"])
        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("LLM Answer:")
            print(result["answer"])
        case "question":
            result = question_command(args.question, args.limit)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("Answer:")
            print(result["answer"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

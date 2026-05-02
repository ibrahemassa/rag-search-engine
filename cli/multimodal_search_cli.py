import argparse

from lib.multimodal_search import search_with_image_command, verify_image_embedding
from lib.augmented_generation import citations_command, question_command, rag_command, summarize_command
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH, RESULTS_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_parser.add_argument("image", type=str, help="Image path",)

    img_serach_parser = subparsers.add_parser("image_search", help="Search using an image")
    img_serach_parser.add_argument("image", type=str, help="Image path",)

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = search_with_image_command(args.image)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['similarity']:.3f})")
                print(res['description'][:DOCUMENT_PREVIEW_LENGTH] + "...\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

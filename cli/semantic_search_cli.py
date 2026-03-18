#!/usr/bin/env python3

import argparse
from lib.chunked_semantic_search import search_chunked_command, semantic_chunk_command
from lib.chunked_semantic_search import embed_chunks
from lib.search_utils import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNKS_SIZE, DEFAULT_MAX_CHUNK_SIZE, RESULTS_LIMIT
from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, search_command, verify_embeddings, verify_model, chunk_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model is loaded")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings are generated")

    query_embed_parse = subparsers.add_parser("embedquery", help="Generate embedding of a query")
    query_embed_parse.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for a query")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, nargs='?', default=RESULTS_LIMIT, help="Results limit")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text to slices of fixed length")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNKS_SIZE, help="Chunks size")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_OVERLAP, help="Count of overlapping words")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk a text")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=DEFAULT_MAX_CHUNK_SIZE, help="Max number of sentences in one chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_OVERLAP, help="Count of overlapping sentences")

    subparsers.add_parser("embed_chunks", help="Generate chunked embeddings")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for a query using chunked semantic search")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=RESULTS_LIMIT, help="Results limit")

    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks()

        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

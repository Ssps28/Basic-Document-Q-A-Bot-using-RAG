"""
main.py — Interactive command-line interface for SmartDocs RAG.

Run with: python -m src.main
"""

from src.query import get_genai_client, get_collection, answer_question
from src.config import COLLECTION_NAME, DB_DIR


def print_banner():
    print("=" * 60)
    print("  SmartDocs RAG — Personal Finance Document Q&A Bot")
    print("=" * 60)
    print("Ask a question about budgeting, saving, credit, retirement,")
    print("or taxes. Type 'exit' or 'quit' to stop.\n")


def main():
    print_banner()

    # Load the client and collection ONCE, outside the loop —
    # re-connecting to ChromaDB on every question would be wasteful.
    try:
        client = get_genai_client()
        collection = get_collection()
    except Exception as e:
        print(f"ERROR: Could not load the vector database from {DB_DIR}.")
        print(f"Did you run 'python -m src.ingest' first? Details: {e}")
        return

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        if not question:
            continue  # ignore empty input, just re-prompt

        print("\nSearching documents and generating answer...\n")

        try:
            result = answer_question(question, client=client, collection=collection)
        except Exception as e:
            print(f"ERROR generating answer: {e}\n")
            continue

        print("-" * 60)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES USED:")
        for citation in result["citations"]:
            print(f"  - {citation}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
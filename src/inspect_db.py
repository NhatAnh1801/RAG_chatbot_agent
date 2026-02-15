"""
Inspect the ChromaDB vector store at ./data/chromadb.

Run from project root: python src/inspect_db.py
Run from src: python inspect_db.py
"""

import os
import chromadb


def get_chroma_path() -> str:
    """Get path to ChromaDB regardless of where script is run from."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "data", "chromadb")


def inspect(
    chroma_path: str | None = None,
    preview_length: int = 300,
    show_embeddings: bool = True,
) -> None:
    """
    Inspect all documents stored in the ChromaDB at the given path.

    Args:
        chroma_path: Path to ChromaDB directory. Defaults to ./data/chromadb.
        preview_length: Max characters to show per document content. Default 300.
        show_embeddings: If True, print embedding vector length (not full vector).
    """
    path = chroma_path or get_chroma_path()

    if not os.path.isdir(path):
        print(f"ChromaDB path not found: {path}")
        return

    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()

    if not collections:
        print("No collections found in the database.")
        return

    for collection in collections:
        count = collection.count()
        print(f"\n{'='*60}")
        print(f"Collection: {collection.name}")
        print(f"Total documents: {count}")
        print(f"{'='*60}")

        if count == 0:
            continue

        include = ["documents", "metadatas"]
        if show_embeddings:
            include.append("embeddings")

        results = collection.get(include=include)

        for i in range(count):
            doc = results["documents"][i] if results["documents"] else None
            meta = results["metadatas"][i] if results["metadatas"] else None
            doc_id = results["ids"][i] if results["ids"] else None

            print(f"\n--- Document {i + 1} (id: {doc_id}) ---")
            if meta:
                for k, v in meta.items():
                    print(f"  {k}: {v}")
            if doc:
                preview = (doc[:preview_length] + "...") if len(doc) > preview_length else doc
                print(f"  Content ({len(doc)} chars): {preview!r}")
            if show_embeddings and len(results["embeddings"]) > 0:
                emb = results["embeddings"][i]
                print(f"  Embedding dim: {emb if emb is not None else 0}")

if __name__ == "__main__":
    inspect()
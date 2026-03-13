"""Offline script to build the ChromaDB index from documents in data/.

Note: This script is for sprint programming purpose.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from src.rag.loader import load_documents
from src.rag.indexer import build_index

DATA_DIR = project_root / "data"

SUBJECT_DIRS = {
    "math": DATA_DIR / "math",
    "chinese": DATA_DIR / "chinese",
}


def main() -> None:
    all_docs = []
    for subject, directory in SUBJECT_DIRS.items():
        if not directory.is_dir() or not any(directory.iterdir()):
            print(f"[SKIP] {directory} — empty or missing")
            continue
        docs = load_documents(directory, subject=subject)
        print(f"[OK]   {subject}: loaded {len(docs)} chunks from {directory}")
        all_docs.extend(docs)

    if not all_docs:
        print("\nNo documents found. Place PDF/MD/TXT files in data/math/ or data/chinese/ first.")
        return

    print(f"\nBuilding index with {len(all_docs)} total chunks ...")
    vectorstore = build_index(all_docs)
    count = vectorstore._collection.count()
    print(f"Index built successfully — {count} vectors in ChromaDB.")


if __name__ == "__main__":
    main()

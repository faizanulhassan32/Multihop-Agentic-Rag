import argparse
import sys
from pathlib import Path

# Add the backend directory (project root) to PYTHONPATH so "src" can be imported
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.ingestion.download_dataset import download_dataset
from src.ingestion.preprocess import preprocess_dataset
from src.ingestion.build_vectorstore import build_vectorstore, load_chunks
from src.ingestion.build_bm25_index import build_bm25_index

def main(sample_size: int = None, force: bool = False):
    # Download (or use cached) dataset
    download_dataset(force=force)
    # Preprocess into chunks
    processed_path, chunks_path = preprocess_dataset(force=force)
    print(f"Processed records: {processed_path}\nChunk file: {chunks_path}")
    # Synchronize both indexes from the same processed chunks.
    chunks = load_chunks()
    chroma_ids = build_vectorstore(force=force, chunks=chunks)
    bm25_ids = build_bm25_index(chunks)
    if chroma_ids != bm25_ids:
        raise RuntimeError(
            "Chroma and BM25 indexes do not contain matching document IDs "
            f"(Chroma: {len(chroma_ids)}, BM25: {len(bm25_ids)})."
        )
    print(f"Ingestion pipeline completed with {len(chroma_ids)} matching chunk IDs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion script for HotpotQA sample.")
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocess, overwriting caches.")
    args = parser.parse_args()
    main(force=args.force)

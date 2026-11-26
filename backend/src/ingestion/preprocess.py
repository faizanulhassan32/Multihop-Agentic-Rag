import json
from pathlib import Path
from typing import List, Dict, Tuple, Set

from config import Config

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_SAMPLE_PATH = PROCESSED_DATA_DIR / "hotpotqa_processed.jsonl"
CHUNKS_PATH = PROCESSED_DATA_DIR / "hotpotqa_chunks.jsonl"


def _normalize_text(text: str) -> str:
    """Normalize whitespace in a string."""
    return " ".join(text.split())


def _article_id(title: str, content: str) -> str:
    """Create a stable ID for a Wikipedia article based on title and content hash."""
    import hashlib
    norm_title = _normalize_text(title).lower()
    norm_content = _normalize_text(content)
    h = hashlib.sha256(f"{norm_title}:{norm_content}".encode()).hexdigest()
    return f"{norm_title}_{h[:8]}"


def preprocess_dataset(raw_path: Path = Path(__file__).resolve().parents[2] / "data" / "raw" / "hotpotqa_sample.jsonl",
                       force: bool = False) -> Tuple[Path, Path]:
    """Read the raw HotpotQA sample and produce:

    * A processed JSONL file preserving the original question records with added metadata.
    * A chunked document JSONL where each line is a dict with keys:
        - `id`: stable document/chunk id
        - `title`: article title
        - `content`: chunk text
        - `metadata`: dict with article and source info
    """
    if PROCESSED_SAMPLE_PATH.exists() and CHUNKS_PATH.exists() and not force:
        print(f"✅ Processed files already exist at {PROCESSED_SAMPLE_PATH} and {CHUNKS_PATH}")
        return PROCESSED_SAMPLE_PATH, CHUNKS_PATH

    print(f"🔧 Preprocessing raw dataset from {raw_path}")
    seen_articles: Set[str] = set()
    chunks: List[Dict] = []
    processed_questions: List[Dict] = []

    # Chunking parameters
    max_chars = Config.CHUNK_SIZE
    overlap = Config.CHUNK_OVERLAP

    with raw_path.open("r", encoding="utf-8") as f_raw:
        for line in f_raw:
            record = json.loads(line)
            processed_questions.append(record)  # keep original record for future eval

            # Context is a dict with parallel "title" and "sentences" lists
            titles = record.get("context", {}).get("title", [])
            sentences_lists = record.get("context", {}).get("sentences", [])
            for title, sentences in zip(titles, sentences_lists):
                # Join sentences preserving order
                article_text = " ".join([_normalize_text(s) for s in sentences])
                article_text = _normalize_text(article_text)
                art_id = _article_id(title, article_text)
                if art_id in seen_articles:
                    continue  # deduplicate identical article content
                seen_articles.add(art_id)

                # Simple sentence-aware splitter based on character length
                start = 0
                while start < len(article_text):
                    end = min(start + max_chars, len(article_text))
                    chunk_text = article_text[start:end]
                    chunk_id = f"{art_id}_chunk_{start}"
                    chunk_meta = {
                        "article_id": art_id,
                        "title": title,
                        "source_dataset": Config.HOTPOT_DATASET,
                        "source_config": Config.HOTPOT_CONFIG,
                        "source_split": Config.HOTPOT_SPLIT,
                        "sentence_count": len(sentences),
                    }
                    chunks.append({"id": chunk_id, "title": title, "content": chunk_text, "metadata": chunk_meta})
                    if end == len(article_text):
                        break
                    start = end - overlap  # overlap for next chunk

    # Write processed questions
    with PROCESSED_SAMPLE_PATH.open("w", encoding="utf-8") as f_out:
        for q in processed_questions:
            json.dump(q, f_out)
            f_out.write("\n")
    print(f"✅ Saved {len(processed_questions)} processed question records to {PROCESSED_SAMPLE_PATH}")

    # Write chunks
    with CHUNKS_PATH.open("w", encoding="utf-8") as f_out:
        for ch in chunks:
            json.dump(ch, f_out)
            f_out.write("\n")
    print(f"✅ Generated {len(chunks)} document chunks to {CHUNKS_PATH}")
    return PROCESSED_SAMPLE_PATH, CHUNKS_PATH

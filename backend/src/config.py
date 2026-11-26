import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env from the backend directory (backend/.env).
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=BACKEND_DIR / ".env")


class Config:
    # Dataset configuration
    HOTPOT_DATASET = os.getenv("HOTPOT_DATASET", "hotpotqa/hotpot_qa")
    HOTPOT_CONFIG = os.getenv("HOTPOT_CONFIG", "distractor")
    HOTPOT_SPLIT = os.getenv("HOTPOT_SPLIT", "train")
    HOTPOT_SAMPLE_SIZE = int(os.getenv("HOTPOT_SAMPLE_SIZE", "500"))
    HOTPOT_RANDOM_SEED = int(os.getenv("HOTPOT_RANDOM_SEED", "42"))

    # Ollama configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CLASSIFIER_MODEL = os.getenv("OLLAMA_CLASSIFIER_MODEL", "qwen2.5:1.5b")
    OLLAMA_EMBEDDING_MODEL = os.getenv(
        "OLLAMA_EMBEDDING_MODEL",
        "nomic-embed-text:latest",
    )
    OLLAMA_GENERATION_MODEL = os.getenv(
        "OLLAMA_GENERATION_MODEL",
        "llama3:latest",
    )

    # Reranker service configuration
    RERANKER_SERVICE_URL = os.getenv(
        "RERANKER_SERVICE_URL",
        "http://127.0.0.1:8010",
    ).strip().rstrip("/")
    RERANKER_REQUEST_TIMEOUT_SECONDS = float(
        os.getenv("RERANKER_REQUEST_TIMEOUT_SECONDS", "30")
    )

    # Chroma configuration
    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(
            BACKEND_DIR
            / "data"
            / "chroma"
            / "hotpotqa_distractor_500"
        ),
    )
    CHROMA_COLLECTION_NAME = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "hotpotqa_distractor_500",
    )

    # BM25 configuration
    BM25_DB_PATH = os.getenv(
        "BM25_DB_PATH",
        str(BACKEND_DIR / "data" / "bm25" / "index.sqlite3"),
    )

    # Hybrid retrieval configuration
    DENSE_RETRIEVAL_K = int(os.getenv("DENSE_RETRIEVAL_K", "10"))
    BM25_RETRIEVAL_K = int(os.getenv("BM25_RETRIEVAL_K", "10"))
    HYBRID_RETRIEVAL_K = int(os.getenv("HYBRID_RETRIEVAL_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))

    # Agent execution configuration
    MAX_PLANNED_TASKS = int(os.getenv("MAX_PLANNED_TASKS", "4"))
    MAX_FOLLOWUP_ROUNDS = int(os.getenv("MAX_FOLLOWUP_ROUNDS", "1"))
    MAX_RETRIEVAL_CONCURRENCY = int(
        os.getenv("MAX_RETRIEVAL_CONCURRENCY", "3")
    )
    MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))
    TASK_RERANK_LIMIT = int(os.getenv("TASK_RERANK_LIMIT", "5"))
    MAX_CONCURRENT_GENERATIONS = int(
        os.getenv("MAX_CONCURRENT_GENERATIONS", "1")
    )

    # Text splitting configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Batch processing
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend directory (backend/.env)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

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
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")


    # Chroma configuration
    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "chroma" / "hotpotqa_distractor_500"),
    )
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hotpotqa_distractor_500")

    # BM25 configuration (independent of the dataset and Chroma paths)
    BM25_DB_PATH = os.getenv(
        "BM25_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "data" / "bm25" / "index.sqlite3"),
    )

    # Hybrid retrieval configuration
    DENSE_RETRIEVAL_K = int(os.getenv("DENSE_RETRIEVAL_K", "10"))
    BM25_RETRIEVAL_K = int(os.getenv("BM25_RETRIEVAL_K", "10"))
    HYBRID_RETRIEVAL_K = int(os.getenv("HYBRID_RETRIEVAL_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))

    # Text splitting configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))  # characters
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Batch processing
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

import sys
from pathlib import Path

# Add project root to PYTHONPATH
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from src.ingestion.config import Config
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def main():
    # Initialise the vectorstore (it will load the persisted collection)
    embedding = OllamaEmbeddings(base_url=Config.OLLAMA_BASE_URL, model=Config.OLLAMA_EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(Path(Config.CHROMA_PERSIST_DIR)),
        embedding_function=embedding,
        collection_name=Config.CHROMA_COLLECTION_NAME,
    )

    # Simple sanity check: number of documents in the collection
    try:
        count = vectorstore._client.count()  # direct client call for count
        print(f"✅ Collection contains {count} documents.")
    except Exception as e:
        print(f"⚠️ Could not retrieve count: {e}")

    # Perform a similarity search for a sample query
    query = "What is the solubility of cadmium chloride in alcohol?"
    results = vectorstore.similarity_search(query, k=3)
    print("\n🔎 Top 3 retrieved documents:")
    for i, doc in enumerate(results, 1):
        print(f"--- Document {i} ---")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Source dataset: {doc.metadata.get('source_dataset')}")
        print(f"Chunk ID: {doc.metadata.get('article_id')}")
        snippet = doc.page_content[:200].replace('\n', ' ')
        print(f"Content snippet: {snippet} ...")
        print()

    # Verify that metadata fields exist for the first document
    if results:
        meta = results[0].metadata
        required_keys = [
            "title",
            "source_dataset",
            "source_config",
            "source_split",
            "sentence_count",
            "article_id",
        ]
        missing = [k for k in required_keys if k not in meta]
        if missing:
            print(f"⚠️ Missing metadata keys: {missing}")
        else:
            print("✅ All expected metadata keys are present in the first retrieved document.")

if __name__ == "__main__":
    main()

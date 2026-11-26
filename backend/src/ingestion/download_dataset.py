import json
import os
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

from config import Config

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_SAMPLE_PATH = RAW_DATA_DIR / "hotpotqa_sample.jsonl"


def download_dataset(force: bool = False) -> Path:
    """Download a sample of the HotpotQA dataset.

    The function respects an existing cached file unless *force* is True.
    It uses streaming to avoid pulling the entire split when possible.
    """
    if RAW_SAMPLE_PATH.exists() and not force:
        print(f"✅ Cached dataset found at {RAW_SAMPLE_PATH}")
        return RAW_SAMPLE_PATH

    print("🔽 Downloading HotpotQA dataset sample…")
    ds = load_dataset(
        Config.HOTPOT_DATASET,
        Config.HOTPOT_CONFIG,
        split=Config.HOTPOT_SPLIT,
        streaming=True,
    )
    # Take a deterministic slice using take after shuffling with a seed if needed
    # Streaming datasets do not support random access; we will simply take the first N samples.
    # For reproducibility we set the seed for the environment (not used here but kept for future).
    os.environ["PYTHONHASHSEED"] = str(Config.HOTPOT_RANDOM_SEED)

    samples: List[Dict] = []
    for i, sample in enumerate(tqdm(ds, total=Config.HOTPOT_SAMPLE_SIZE, desc="samples")):
        if i >= Config.HOTPOT_SAMPLE_SIZE:
            break
        samples.append(sample)

    # Write to JSONL
    with RAW_SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            json.dump(s, f)
            f.write("\n")
    print(f"✅ Saved {len(samples)} samples to {RAW_SAMPLE_PATH}")
    return RAW_SAMPLE_PATH

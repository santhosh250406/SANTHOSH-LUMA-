import os
import json
import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directory that contains your knowledge base text files
KB_FOLDER = "kb"
EMBEDDINGS_FILE = os.path.join("rag", "kb_index.json")

# Initialize model (you can change this if you have another embedding model)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL_NAME)


def load_kb_texts(kb_folder: str = KB_FOLDER) -> Tuple[List[str], List[str]]:
    """
    Loads all text files from the KB folder and returns their content and file names.
    """
    if not os.path.exists(kb_folder):
        logger.warning(f"KB folder '{kb_folder}' not found. Creating one.")
        os.makedirs(kb_folder, exist_ok=True)
        return [], []

    texts, paths = [], []
    for filename in os.listdir(kb_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
                    paths.append(file_path)
                    logger.info(f"Loaded KB file: {filename} ({len(content.split())} words)")
    return texts, paths


def build_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for each KB text using a sentence-transformer model.
    """
    if not texts:
        logger.warning("No KB texts found to build embeddings.")
        return np.array([])
    logger.info("Generating embeddings for KB texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def save_index(embeddings: np.ndarray, kb_texts: List[str], kb_paths: List[str]):
    """
    Saves embeddings and corresponding KB metadata to JSON for retrieval.
    """
    data = {
        "texts": kb_texts,
        "paths": kb_paths,
        "embeddings": embeddings.tolist(),
    }
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info(f"Saved KB index to {EMBEDDINGS_FILE}")


def load_index() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Loads the embeddings index file if it exists.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        logger.warning("KB index not found. Run build_index() first.")
        return np.array([]), [], []

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array(data["embeddings"])
    texts = data["texts"]
    paths = data["paths"]
    logger.info(f"Loaded KB index with {len(texts)} entries.")
    return embeddings, texts, paths


def build_index() -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Main function: loads KB, builds embeddings, and saves index.
    Returns embeddings, texts, and paths.
    """
    kb_texts, kb_paths = load_kb_texts()
    if not kb_texts:
        logger.warning("No KB files found. Skipping index build.")
        return np.array([]), [], []

    embeddings = build_embeddings(kb_texts)
    save_index(embeddings, kb_texts, kb_paths)
    return embeddings, kb_texts, kb_paths


if __name__ == "__main__":
    # Run manually if needed
    logger.info("Building knowledge base index...")
    build_index()
    logger.info("✅ KB index built successfully.")

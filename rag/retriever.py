from sentence_transformers import SentenceTransformer
import faiss
import os, numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(kb_folder="kb"):
    texts, paths = [], []
    for f in os.listdir(kb_folder):
        with open(os.path.join(kb_folder, f), "r", encoding="utf-8") as file:
            texts.append(file.read())
            paths.append(f)
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts, paths

index, kb_texts, kb_paths = build_index()

def retrieve_relevant(user_query, top_k=2):
    query_emb = model.encode([user_query])
    D, I = index.search(np.array(query_emb), top_k)
    results = [kb_texts[i] for i in I[0]]
    return results

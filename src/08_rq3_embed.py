"""Step 8: sentence embeddings of the RQ3 documents, for the BERTopic triangulation (step 9).

Embeddings are computed from the cleaned prose of the same 3,816 documents that LDA uses
(04_rq3_preprocess.py, column `prose`), with sentence-transformers all-MiniLM-L6-v2 at a
pinned revision (Reimers & Gurevych, 2019). The model reads at most 256 word pieces, so
long issues are split into 150-word chunks; the chunk embeddings are averaged (mean
pooling) and re-normalised, so each embedding represents the whole document, as in LDA.

Why a separate script: the encoder runs fastest multi-threaded, whereas step 9 must run with a
single OpenMP thread (see its header), so the two steps run in separate processes.

Reads data/processed/rq3_corpus.csv. Writes data/processed/rq3_embeddings_chunked.npy
(not redistributed; it is derived from the issue text).
Run:  python src/08_rq3_embed.py      (about 5-10 minutes on a laptop CPU; downloads the model once)
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import PROCESSED

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pinned Hugging Face snapshot
CHUNK_WORDS = 150  # below the 256 word-piece limit


def main():
    c = pd.read_csv(PROCESSED / "rq3_corpus.csv")
    chunks, owner = [], []
    for i, text in enumerate(c.prose.fillna("")):
        words = text.split() or [""]
        for k in range(0, len(words), CHUNK_WORDS):
            chunks.append(" ".join(words[k : k + CHUNK_WORDS]))
            owner.append(i)
    encoder = SentenceTransformer(EMB_MODEL, revision=EMB_REVISION, device="cpu")
    chunk_emb = encoder.encode(chunks, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    owner = np.array(owner)
    emb = np.vstack([chunk_emb[owner == i].mean(0) for i in range(len(c))])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(PROCESSED / "rq3_embeddings_chunked.npy", emb)
    print(f"{len(c)} documents, {len(chunks)} chunks -> embeddings {emb.shape}")


if __name__ == "__main__":
    main()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
CAP = 1500
def place_similarity(texts, seed=0):
    """Per-place TF-IDF + nearest-neighbour cosine similarity. Returns sims, neighbours, sampled flag."""
    n = len(texts); sampled = n > CAP
    pool = np.sort(np.random.default_rng(seed).choice(n, CAP, replace=False)) if sampled else np.arange(n)
    if len(pool) < 3: return {}, {}, sampled
    V = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english").fit_transform([texts[i] for i in pool])
    k = min(4, len(pool)); d, ix = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(V).kneighbors(V)
    sims = {int(pool[i]): float(1 - d[i, 1]) for i in range(len(pool))}
    nb = {int(pool[i]): [(int(pool[ix[i, j]]), float(1 - d[i, j])) for j in range(1, k)] for i in range(len(pool))}
    return sims, nb, sampled

import numpy as np, pandas as pd
from .feature_engineering import build
from .model import fit
T1, T2 = 40, 65
ACTIVE = {"text_similarity": .8, "review_burst": 5, "rating_concentration": .3, "reviewer_activity": 5}
def status(s): return "normal" if s < T1 else "needs_review" if s < T2 else "suspicious"
def run(df, y):
    X, notes, sampled, nb = build(df); model, s, info = fit(X, y)
    act = sum((X[k] >= t).astype(int) for k, t in ACTIVE.items() if k in X)
    s = np.where(act < 2, np.minimum(s, T1 - 1), s)   # fairness guard: one signal alone never raises a flag
    df = df.assign(score=np.round(s, 1), status=[status(v) for v in s], signals=act.values)
    places = []
    for i, (name, g) in enumerate(sorted(df.groupby("place"), key=lambda kv: -len(kv[1]))):
        top = g.score.nlargest(max(3, len(g) // 10)); sc = round(float(top.mean()), 1) if len(g) >= 3 else 0.0
        places.append({"id": f"p{i}", "name": name, "reviews": len(g), "avg_rating": round(float(g.rating.mean()), 2) if g.rating.notna().any() else None, "score": sc, "status": status(sc), "idx": g.index.tolist()})
    return {"df": df, "X": X, "model": model, "info": info, "notes": notes, "sampled": sampled, "nb": nb, "places": places}

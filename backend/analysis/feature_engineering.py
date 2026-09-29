import numpy as np, pandas as pd
from .text_similarity import place_similarity
LABELS = {"text_similarity": "Review-text similarity", "review_burst": "Review burst", "rating_concentration": "Rating concentration",
 "min_gap_minutes": "Time gap between reviews", "reviewer_activity": "Reviewer activity", "reviewer_places": "Places per reviewer", "review_length": "Review length"}
WIN = 10.0
def build(df):
    X = pd.DataFrame(index=df.index); notes = []; sampled = False; nb = {}
    X["review_length"] = df.review.str.split().str.len().astype(float)
    sim = pd.Series(0.0, index=df.index)
    for _, g in df.groupby("place"):
        s, n, sp = place_similarity(g.review.tolist()); sampled |= sp
        loc = g.index.values; sim.loc[loc[list(s)]] = list(s.values()) if s else []
        nb.update({int(loc[a]): [(int(loc[b]), v) for b, v in l] for a, l in n.items()})
    X["text_similarity"] = sim
    if df.date.notna().any():
        mins = (df.date - df.date.min()).dt.total_seconds() / 60
        for c in ["review_burst", "min_gap_minutes", "rating_concentration"]: X[c] = 0.0
        X["min_gap_minutes"] = 1e5
        for _, g in df[df.date.notna()].groupby("place"):
            o = g.index[np.argsort(mins[g.index].values)]; t = mins[o].values
            X.loc[o, "review_burst"] = np.searchsorted(t, t + WIN, "right") - np.searchsorted(t, t - WIN, "left") - 1
            if len(t) > 1:
                dg = np.diff(t); X.loc[o, "min_gap_minutes"] = np.minimum(np.r_[1e5, dg], np.r_[dg, 1e5]).clip(0, 1e4)
            if df.rating.notna().any():
                r = df.rating[o].values; tot = np.searchsorted(t, t + 1440, "right") - np.searchsorted(t, t - 1440, "left"); base = pd.Series(r).map(pd.Series(r).value_counts(normalize=True)).values
                same = np.array([((np.abs(t - t[i]) <= 1440) & (r == r[i])).sum() for i in range(len(t))]) if len(t) < 4000 else tot * base
                X.loc[o, "rating_concentration"] = np.where(tot >= 5, np.clip(same / tot - base, 0, 1), 0)
    else: notes.append("Time analysis (bursts, gaps) unavailable because no valid timestamp column is present.")
    if df.reviewer.notna().any():
        X["reviewer_activity"] = df.groupby("reviewer")["review"].transform("count").fillna(1).astype(float)
        X["reviewer_places"] = df.groupby("reviewer")["place"].transform("nunique").fillna(1).astype(float)
    else: notes.append("Reviewer behavior analysis unavailable because reviewer ID is not present.")
    if not df.rating.notna().any(): notes.append("Rating analysis unavailable because no rating column is present.")
    return X.fillna(0), notes, sampled, nb

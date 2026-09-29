import io, numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analysis.preprocessing import detect_columns, standardize, demo_df
from analysis.scoring import run
from analysis.explainability import shap_values, contributions, sentence
app = FastAPI(title="TrustPlate"); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = {"raw": None, "res": None, "demo": False}
def res():
    if not S["res"]: raise HTTPException(400, "No analysis yet. Upload a dataset or load the demo data first.")
    return S["res"]
def dt(v): return None if pd.isna(v) else str(v)
@app.get("/health")
def health(): return {"ok": True}
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try: df = pd.read_csv(io.BytesIO(await file.read()), encoding_errors="replace", low_memory=False)
    except Exception as e: raise HTTPException(400, f"Could not read this file as CSV: {e}")
    S["raw"] = df; S["demo"] = False
    return {"columns": list(df.columns), "mapping": detect_columns(df), "rows": len(df)}
@app.post("/analyze")
def analyze(body: dict):
    try:
        if body.get("demo"): df = demo_df(); S["demo"] = True; m = detect_columns(df)
        else:
            if S["raw"] is None: raise ValueError("Upload a CSV first.")
            df = S["raw"]; m = body.get("mapping") or detect_columns(df); S["demo"] = False
        d, y = standardize(df, {k: (v or None) for k, v in m.items()})
        if len(d) < 5: raise ValueError("Need at least 5 non-empty reviews.")
        S["res"] = run(d, y)
    except ValueError as e: raise HTTPException(400, str(e))
    return summary()
@app.get("/summary")
def summary():
    r = res(); d = r["df"]; c = d.status.value_counts()
    act = None
    if d.date.notna().any(): act = [{"date": str(k), "reviews": int(v)} for k, v in d.dropna(subset=["date"]).groupby(d.date.dt.to_period("W").astype(str)).size().items()]
    return {"total": len(d), "places": len(r["places"]), "counts": {k: int(c.get(k, 0)) for k in ["normal", "needs_review", "suspicious"]}, "demo": S["demo"], "info": r["info"], "notes": r["notes"], "sampled": r["sampled"],
      "ratings": [{"rating": str(k), "count": int(v)} for k, v in d.rating.dropna().round().value_counts().sort_index().items()], "activity": act}
@app.get("/places")
def places(): return [{k: v for k, v in p.items() if k != "idx"} for p in res()["places"]]
def rev(r, i, full=False):
    x = r["df"].loc[i]; o = {"id": f"r{i}", "place": x.place, "rating": None if pd.isna(x.rating) else float(x.rating), "review": x.review if full else x.review[:140], "reviewer": x.reviewer, "date": dt(x.date), "score": float(x.score), "status": x.status}
    return o
@app.get("/places/{pid}")
def place(pid: str):
    r = res(); p = next((p for p in r["places"] if p["id"] == pid), None)
    if not p: raise HTTPException(404, "Place not found")
    g = r["df"].loc[p["idx"]]; X = r["X"].loc[p["idx"]]
    ev = [f"{int((X.text_similarity >= .8).sum())} reviews showed high textual similarity to another review at this place."]
    if "review_burst" in X: ev.append(f"Up to {int(X.review_burst.max()) + 1} reviews appeared within a 10-minute period.")
    if "rating_concentration" in X: ev.append(f"Peak same-rating concentration in a 24-hour window was {X.rating_concentration.max():.0%} above this place's usual share.")
    neg = g[(g.rating <= 2) & (g.status == "normal")].head(3).index
    return {**{k: v for k, v in p.items() if k != "idx"}, "evidence": ev, "model": r["info"], "analyzed": len(g), "sampled": r["sampled"],
      "top": [rev(r, i) for i in g.score.nlargest(5).index], "negatives": [rev(r, i) for i in neg]}
@app.get("/reviews")
def reviews(place: str = "", rating: float = None, status: str = "", min_score: float = 0, date_from: str = "", date_to: str = "", limit: int = 200):
    r = res(); d = r["df"]; m = d.score >= min_score
    if place: m &= d.place == place
    if rating is not None: m &= d.rating.round() == rating
    if status: m &= d.status == status
    if date_from: m &= d.date >= pd.Timestamp(date_from)
    if date_to: m &= d.date <= pd.Timestamp(date_to) + pd.Timedelta(days=1)
    idx = d[m].sort_values("score", ascending=False).index[:limit]
    return {"total": int(m.sum()), "items": [rev(r, i) for i in idx]}
@app.get("/explanation/{eid}")
def explanation(eid: str):
    r = res(); X = r["X"]; mode = r["info"]["mode"]
    if eid.startswith("r"):
        i = int(eid[1:]); sv, meth = shap_values(r["model"], X.loc[[i]], X, mode); cs = contributions(sv[0], X.columns)
        sim = [{**rev(r, j), "similarity": round(s, 3)} for j, s in r["nb"].get(i, []) if s > .3]
        out = {"kind": "review", "review": rev(r, i, True), "similar": sim, "signals": int(r["df"].loc[i].signals)}
    else:
        p = next((p for p in r["places"] if p["id"] == eid), None)
        if not p: raise HTTPException(404, "Not found")
        top = r["df"].loc[p["idx"]].score.nlargest(min(30, len(p["idx"]))).index; sv, meth = shap_values(r["model"], X.loc[top], X, mode); cs = contributions(sv.mean(0), X.columns)
        pl = place(eid); out = {"kind": "place", "place": {k: pl[k] for k in ["id", "name", "score", "status", "reviews"]}, "evidence": pl["evidence"]}
    stt = out.get("review", out.get("place"))["status"]
    return {**out, "method": meth, "contributions": cs, "text": sentence(cs), "recommendation": "Human verification recommended." if stt != "normal" else "No action needed. Treat this as a legitimate consumer opinion."}

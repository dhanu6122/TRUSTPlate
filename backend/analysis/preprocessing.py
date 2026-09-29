import numpy as np, pandas as pd
ALIASES = {"review": ["review", "review_text", "text", "comment", "content"], "rating": ["rating", "stars", "score"],
 "place": ["restaurant", "restaurant_name", "business_name", "place", "place_id"], "reviewer": ["user_id", "reviewer_id", "author", "username"],
 "date": ["date", "review_date", "timestamp", "time"], "label": ["label", "is_fake", "fake", "deceptive", "class", "target", "suspicious"]}
POS = {"1", "1.0", "true", "yes", "fake", "deceptive", "suspicious"}; NEG = {"0", "0.0", "false", "no", "genuine", "truthful", "normal", "non-deceptive"}

def detect_columns(df):
    low = {str(c).lower().strip(): c for c in df.columns}
    return {k: next((low[a] for a in v if a in low), None) for k, v in ALIASES.items()}

def standardize(df, m):
    if not m.get("review") or m["review"] not in df.columns:
        raise ValueError("A review text column is required. Map one in the column mapping step.")
    col = lambda k: df[m[k]].reset_index(drop=True) if m.get(k) and m[k] in df.columns else None
    out = pd.DataFrame({"review": col("review").fillna("").astype(str)})
    out["rating"] = pd.to_numeric(col("rating"), errors="coerce") if col("rating") is not None else np.nan
    out["place"] = col("place").fillna("Unknown").astype(str) if col("place") is not None else "All reviews"
    out["reviewer"] = col("reviewer").astype(str).where(col("reviewer").notna()) if col("reviewer") is not None else None
    out["date"] = pd.to_datetime(col("date"), errors="coerce") if col("date") is not None else pd.NaT
    y = None
    if col("label") is not None:
        s = col("label").astype(str).str.lower().str.strip().map(lambda v: 1 if v in POS else 0 if v in NEG else np.nan)
        if s.notna().mean() > .9 and s.nunique() == 2: y = s.fillna(0).astype(int)
    keep = out.review.str.strip() != ""
    out = out[keep].reset_index(drop=True)
    return out, (y[keep].reset_index(drop=True) if y is not None else None)

def demo_df():
    r = np.random.default_rng(7); rows = []
    pos = ["Excellent food and friendly service", "Loved the biryani, generous portions", "Great ambience and quick service", "Tasty and fresh, will come again", "Good value for money, lovely staff"]
    neg = ["Food arrived cold and delivery was late", "Waited 50 minutes and the order was wrong", "Too salty and the portion was small", "Rude staff and a dirty table"]
    mid = ["Decent food but a bit pricey", "Average taste, service was slow", "Okay for a quick lunch"]
    dish = ["dosa", "biryani", "paneer tikka", "noodles", "soup", "salad", "kebab", "thali", "pasta", "burger", "curry", "dessert"]
    base = pd.Timestamp("2026-06-01")
    for p in ["Chennai Spice Restaurant", "Harbor Grill", "Green Bowl Cafe", "Urban Tandoor"]:
        for _ in range(90):
            k = r.choice(3, p=[.6, .15, .25]); t = [pos, neg, mid][k]; rt = [int(r.choice([4, 5])), int(r.choice([1, 2])), 3][k]
            rows.append((p, f"u{r.integers(1, 400)}", rt, f"{r.choice(t)}. The {r.choice(dish)} was {r.choice(['good', 'fine', 'the best part', 'memorable'])} on {r.choice(['Friday', 'Sunday', 'Tuesday'])}.", base + pd.Timedelta(minutes=int(r.integers(0, 129600)))))
    for i in range(18):
        rows.append(("Chennai Spice Restaurant", f"s{i % 6}", 5, f"Amazing food best restaurant in the city must visit five stars{'!' * (i % 3 + 1)}", pd.Timestamp("2026-08-14 20:00") + pd.Timedelta(minutes=int(r.integers(0, 10)))))
    for i in range(6):
        rows.append(("Urban Tandoor", f"n{i}", 5, "Wonderful place highly recommended for family dinner, superb taste", base + pd.Timedelta(days=10 + i * 3, hours=int(r.integers(0, 24)))))
    return pd.DataFrame(rows, columns=["restaurant_name", "user_id", "rating", "review_text", "date"])

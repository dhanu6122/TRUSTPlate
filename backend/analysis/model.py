import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .anomaly_detection import isolation
def fit(X, y):
    if y is not None and y.value_counts().min() >= 5:
        a, b, ya, yb = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
        m = RandomForestClassifier(200, class_weight="balanced", random_state=42, n_jobs=-1).fit(a, ya); p = m.predict(b)
        return m, m.predict_proba(X)[:, 1] * 100, {"mode": "supervised", "model": "Random Forest", "test_size": len(b), "accuracy": accuracy_score(yb, p),
          "precision": precision_score(yb, p, zero_division=0), "recall": recall_score(yb, p, zero_division=0), "f1": f1_score(yb, p, zero_division=0)}
    m, s, n = isolation(X)
    return m, s, {"mode": "unsupervised", "model": "Isolation Forest", "records": len(X), "anomalies": n, "anomaly_pct": round(100 * n / len(X), 2)}

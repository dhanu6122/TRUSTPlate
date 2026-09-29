from sklearn.ensemble import IsolationForest
import numpy as np
def isolation(X):
    m = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(X)
    score = np.clip((-m.score_samples(X) - .4) / .25, 0, 1) * 100
    return m, score, int((m.predict(X) == -1).sum())

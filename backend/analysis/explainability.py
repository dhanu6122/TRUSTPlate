import numpy as np
from .feature_engineering import LABELS
def shap_values(model, Xs, Xall, mode):
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(Xs)
        if isinstance(sv, list): sv = sv[1]
        sv = np.asarray(sv); sv = sv[:, :, 1] if sv.ndim == 3 else sv
        return (-sv if mode == "unsupervised" else sv), "SHAP (TreeExplainer)"
    except Exception:
        z = ((Xs - Xall.mean()) / (Xall.std() + 1e-9)).clip(-3, 3).values * .05
        return z, "Standardised deviation (SHAP unavailable)"
def contributions(sv, cols):
    return sorted([{"feature": c, "label": LABELS[c], "value": round(float(v), 4)} for c, v in zip(cols, sv)], key=lambda d: -abs(d["value"]))
def sentence(cs):
    top = [c["label"].lower() for c in cs if c["value"] > .01][:2]
    if not top: return "No feature pushed the model's assessment toward suspicion. This looks like ordinary review activity."
    return f"{' and '.join(top).capitalize()} {'was' if len(top) == 1 else 'were'} the strongest contributor{'' if len(top) == 1 else 's'} to the model's prediction. This describes a pattern in the data, not proof of wrongdoing."

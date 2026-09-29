# TrustPlate — Explainable AI for Trusted Reviews
*Detect Patterns. Explain Decisions. Protect Genuine Opinions.*

## Overview, problem and solution
Problem: identify suspicious review patterns while preserving legitimate opinions and avoiding unjustified accusations.
Solution: TrustPlate scores *patterns* per review and per place, explains each score with SHAP, and leaves the decision to a human. It never says "fake".

## Architecture
React + TypeScript + Vite + Tailwind + Recharts (frontend) ⇄ FastAPI (backend: pandas, scikit-learn, SHAP). No external or paid APIs. State is held in backend memory.

## AI methodology
- **Feature engineering** (only from available columns): per-place TF-IDF nearest-neighbour text similarity, review length, review bursts (reviews within ±10 min), gap to neighbouring reviews, same-rating concentration in a 24h window relative to the place's usual share, reviewer review count and places per reviewer. Missing columns are skipped with a visible notice.
- **Model:** labels found (`label`, `is_fake`, `deceptive`…) → Random Forest with a stratified split; Accuracy/Precision/Recall/F1 are computed on the held-out set. No labels → Isolation Forest; only record and anomaly counts are shown.
- **SHAP:** `TreeExplainer` per review, and averaged over a place's top-scoring reviews. Red bars raise suspicion, teal bars lower it. SHAP explains the model, it does not prove anything.
- **Categories:** <40 Normal, 40–64 Needs review, ≥65 Suspicious pattern.
- **Fairness:** rating value and deviation are *not* features, so a negative or positive rating alone cannot raise suspicion. A safeguard caps any review with fewer than two unusual signals below the review threshold. No demographic or personal attributes are used.
- **Scale:** comparison is per place (never all-to-all); places with >1,500 reviews are sampled for text similarity and the dashboard says so.

## Dataset requirements
CSV with at least a review text column. Optional: rating, place, reviewer, date, label. Recognised names: review/review_text/text/comment/content; rating/stars/score; restaurant/restaurant_name/business_name/place/place_id; user_id/reviewer_id/author/username; date/review_date/timestamp/time. Example:
```
restaurant_name,user_id,rating,review_text,date
Chennai Spice,u12,1,"Food arrived cold and delivery was late",2026-06-03 19:42
```

## Installation (Windows, PowerShell)
```
cd trustplate\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
New terminal:
```
cd trustplate\frontend
npm install
npm run dev
```
Open http://localhost:5173.

## Upload a Kaggle CSV
Dashboard → **Upload dataset** → check the detected mapping (change with the dropdowns) → **Run AI analysis**. Or click **Load demo data**.

## Limitations
In-memory single-user state; heuristic thresholds; unlabeled mode finds unusual, not confirmed, activity; text similarity is lexical (TF-IDF).

## Future improvements
Sentence embeddings, persistent storage, reviewer graph analysis, audit log for human decisions.

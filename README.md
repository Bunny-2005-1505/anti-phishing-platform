# Anti-Phishing Detection and Awareness Platform

Real-time, explainable phishing URL detection built for the capstone
"Anti-Phishing Detection and Awareness Platform" (VIT-AP, SCOPE). Every
verdict is paired with the specific features that drove it, in plain
language, following the project's objective of combining detection with an
"awareness moment" at the point of use.

## Pipeline

```
URL input → feature extraction → ML classifier → explanation generator → verdict + awareness cue
```

- **Feature extraction** (`feature_extraction.py`) — 20 lightweight,
  fast-to-compute lexical/structural features derived from the URL string
  alone (length, hyphen/digit/dot counts, IP-literal host, HTTPS usage,
  subdomain depth, URL-shortener use, suspicious keywords, suspicious TLD,
  etc.). No page fetch required, so it's safe to run at low latency.
- **Classifier** (`train_model.py`) — trains Random Forest and Gradient
  Boosting classifiers and keeps whichever scores higher on F1, following
  the classical-ML baseline in Alsarhan, Igried & Alauthman (2023).
- **Explanation** (`explain.py`) — uses SHAP's TreeExplainer when `shap`
  is installed for proper per-instance attribution; otherwise falls back to
  a transparent heuristic (global feature importance × how far this URL's
  value sits from the typical legitimate/phishing value) so the app still
  works with zero extra dependencies.
- **App** (`app.py`) — Streamlit UI with three pages: **Detector** (check a
  URL, see the verdict, the top contributing features in plain language,
  and awareness tips), **Model Performance** (accuracy/precision/recall/F1/
  ROC-AUC/false-positive rate/latency, confusion matrix, feature
  importance chart), and **About**.

## ⚠️ About the training data — read before your final submission

This was built in a sandboxed environment with no internet access, so the
real dataset referenced in the proposal (UCI "Phishing Websites" dataset /
PhishTank feed) could not be downloaded. `dataset_generator.py` instead
generates a **synthetic** dataset whose per-feature distributions are
modeled on the patterns reported in the literature (phishing URLs tend to
be longer, use more hyphens/digits/suspicious words, use IP hosts or
shorteners more often, use HTTPS less often, etc.). This lets the whole
pipeline run end-to-end out of the box, but the ~99.8% accuracy you'll see
on the Model Performance page reflects how separable the *synthetic* data
is, not real-world performance — treat it as a placeholder, not a result to
cite.

**To swap in real data:**

1. Download the UCI Phishing Websites dataset
   (https://archive.ics.uci.edu/dataset/327/phishing+websites) or a
   PhishTank/Kaggle CSV of raw phishing + legitimate URLs.
2. If your source gives raw URLs rather than pre-extracted features, run
   each one through `feature_extraction.extract_features()` to build a
   table with the same 20 columns as `FEATURE_NAMES` in
   `feature_extraction.py`, plus a `label` column (1 = phishing,
   0 = legitimate).
3. Save it as `data/real_dataset.csv` and change the one line in
   `train_model.py` that calls `generate_dataset(...)` to instead
   `pd.read_csv("data/real_dataset.csv")`.
4. Re-run `python train_model.py` — it regenerates
   `models/model.pkl`, `metrics.json`, `feature_importance.json`, and
   `train_stats.json` automatically; nothing else needs to change.

## Running locally

```bash
pip install -r requirements.txt
python train_model.py   # only needed once, or after changing the dataset — model.pkl is already included
streamlit run app.py
```

## Deploying (e.g. Streamlit Community Cloud, like your reference app)

1. Push this folder to a GitHub repo.
2. On https://share.streamlit.io, create a new app pointing at `app.py`
   in that repo.
3. Streamlit Cloud will install `requirements.txt` automatically —
   `shap` will then be available there even though this sandbox couldn't
   install it, so you'll automatically get proper SHAP-based explanations
   instead of the heuristic fallback.

## Extending toward the full proposal

The current build covers the URL-structure detection + explanation loop.
Natural next steps to fully match the proposal:

- **Page/HTML-structure features** — fetch the page (where safe to do so)
  and add features like form-action mismatches, favicon domain mismatch,
  external-resource ratio, etc.
- **Domain-info features** — WHOIS domain age, TLS certificate validity —
  need live network access at inference time; wrap in try/except with a
  neutral fallback value since WHOIS lookups can be rate-limited or blocked.
- **Retraining feedback loop** — periodically retrain on a fresh PhishTank
  sample to address concept drift, per the architecture diagram.
- **Evaluation baselines** — the proposal calls for comparing against a
  black-box classifier (no explanation) and a rule-based filter; the
  Model Performance page currently only reports the proposed model, so add
  those two baselines for the final comparison table.

## File overview

```
app.py                    Streamlit UI (3 pages: Detector, Model Performance, About)
feature_extraction.py     URL -> 20-feature dict
dataset_generator.py      Synthetic training data generator (see warning above)
train_model.py            Trains RF + GB, saves best model + metrics + explanations data
explain.py                Per-verdict explanation generator (SHAP or heuristic fallback)
models/model.pkl          Pre-trained classifier (ships ready to run)
models/metrics.json       Evaluation metrics used by the Model Performance page
models/feature_importance.json   Global feature importances
models/train_stats.json   Per-class per-feature mean/std, used for explanation sentences
requirements.txt
```

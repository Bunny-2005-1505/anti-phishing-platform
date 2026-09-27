
import time

import joblib
import pandas as pd
import streamlit as st

from explain import Explainer, get_awareness_tips
from feature_extraction import FEATURE_NAMES, extract_features

st.set_page_config(
    page_title="Anti-Phishing Detection & Awareness Platform",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    explainer = Explainer(model)
    return model, explainer


@st.cache_data
def load_metrics():
    import json
    with open("models/metrics.json") as f:
        return json.load(f)


def verdict_card(is_phishing: bool, confidence: float):
    if is_phishing:
        st.markdown(
            f"""
            <div style="padding:1.25rem;border-radius:0.6rem;background:#3b1010;
                        border:1px solid #7a1f1f;">
              <h2 style="color:#ff6b6b;margin:0;">⚠️ Likely Phishing</h2>
              <p style="color:#f2b8b8;margin:0.3rem 0 0 0;">
                Confidence: {confidence*100:.1f}%
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="padding:1.25rem;border-radius:0.6rem;background:#0f2f1c;
                        border:1px solid #1f7a3f;">
              <h2 style="color:#5ce08a;margin:0;">✅ Looks Legitimate</h2>
              <p style="color:#b8f2cd;margin:0.3rem 0 0 0;">
                Confidence: {confidence*100:.1f}%
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_detector(model, explainer):
    st.title("🛡️ Anti-Phishing Detection & Awareness Platform")
    st.caption(
        "Real-time, explainable phishing URL detection — every verdict comes "
        "with the reasons behind it, so each check is also a quick awareness lesson."
    )

    url = st.text_input(
        "Enter a URL to check",
        placeholder="e.g. https://secure-paypal-login.tk/update-account",
    )
    col_a, col_b = st.columns([1, 5])
    analyze = col_a.button("Analyze", type="primary", use_container_width=True)

    if analyze and url.strip():
        t0 = time.perf_counter()
        feats = extract_features(url)
        vec = pd.DataFrame([[feats[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
        proba = model.predict_proba(vec)[0]
        is_phishing = proba[1] >= 0.5
        confidence = proba[1] if is_phishing else proba[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        verdict_card(is_phishing, confidence)
        st.caption(f"Checked in {latency_ms:.1f} ms")

        st.subheader("Why this verdict — top contributing features")
        top_features = explainer.explain(feats, top_k=5)
        for entry in top_features:
            icon = "🚩" if entry["direction"] == "phishing" else "✅"
            st.markdown(f"{icon} {entry['sentence']}")

        st.subheader("Awareness cues for next time")
        for tip in get_awareness_tips(top_features):
            st.info(tip)

        with st.expander("Show all extracted features"):
            st.dataframe(
                pd.DataFrame([feats]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )
    elif analyze:
        st.warning("Please enter a URL first.")


def page_performance(metrics):
    st.title("📊 Model Performance")
    st.caption(
        "Evaluated on a held-out test split. See README for the note on the "
        "synthetic training data used in this build — swap in the real UCI "
        "Phishing Websites dataset before final submission."
    )

    best = metrics["best_model"]
    res = metrics["all_results"][best]
    st.markdown(f"**Selected model:** `{best}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{res['precision']*100:.2f}%")
    c3.metric("Recall", f"{res['recall']*100:.2f}%")
    c4.metric("F1-score", f"{res['f1']*100:.2f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("ROC-AUC", f"{res['roc_auc']:.4f}")
    c6.metric("False Positive Rate", f"{res['false_positive_rate']*100:.2f}%")
    c7.metric("Inference latency", f"{res['inference_latency_ms']:.3f} ms/URL")

    st.subheader("Model comparison")
    rows = []
    for name, r in metrics["all_results"].items():
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC-AUC": r["roc_auc"],
            "FPR": r["false_positive_rate"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.subheader("Confusion matrix (best model)")
    cm = res["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        index=["Actual: Legitimate", "Actual: Phishing"],
        columns=["Predicted: Legitimate", "Predicted: Phishing"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Global feature importance")
    import json
    with open("models/feature_importance.json") as f:
        importances = json.load(f)
    imp_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
    st.bar_chart(imp_df.set_index("Feature"))


def page_about():
    st.title("ℹ️ About this platform")
    st.markdown(
        """
This project follows a **detection + awareness** design: instead of returning a
bare *phishing / safe* label, every verdict is paired with the specific features
that drove it, in plain language — turning each check into a small security
lesson.

**Pipeline:** URL input → lightweight feature extraction (URL structure only,
no page fetch needed) → classical ML classifier (Random Forest / Gradient
Boosting) → per-verdict feature attribution → plain-language explanation +
awareness cue.

**Base papers**
- PhishFind — Mendoza Vega, Diaz Mercado, Escobedo (2025): real-time
  ML-based detection pipeline template.
- Alsarhan, Igried & Alauthman (2023) — *Enhancing Phishing URL Detection: A
  Comparative Study of Machine Learning Algorithms*: classical-ML baseline
  (~94.6% accuracy) this platform builds its classifier and explanation
  layer on.

**Limitations to keep in mind**
- Detection is based on the URL string only, so it won't catch phishing
  pages hosted on otherwise-clean-looking legitimate domains (e.g. a
  compromised WordPress site). Extending feature extraction to page
  content/HTML structure is the natural next step.
- The bundled model is trained on a synthetic dataset (see README) —
  replace it with a real phishing-URL dataset before treating results as
  representative of real-world accuracy.
- Concept drift: like any ML detector, this needs periodic retraining as
  phishing tactics evolve.
        """
    )


def main():
    model, explainer = load_model()
    metrics = load_metrics()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Detector", "Model Performance", "About"])

    if page == "Detector":
        page_detector(model, explainer)
    elif page == "Model Performance":
        page_performance(metrics)
    else:
        page_about()


if __name__ == "__main__":
    main()

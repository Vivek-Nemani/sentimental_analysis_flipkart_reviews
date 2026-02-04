import pickle
from pathlib import Path

import numpy as np
import streamlit as st


st.set_page_config(page_title="Flipkart Sentiment Analyzer")

MODEL_PATH = Path("models/model.pkl")
VECTORIZER_PATH = Path("models/vectorizer.pkl")


@st.cache_resource(show_spinner=False)
def load_artifact(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_logreg_defaults(model):
    """Ensure pickled LogisticRegression has expected attrs when loaded across sklearn versions."""
    if model is None:
        return None
    if model.__class__.__name__ == "LogisticRegression":
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"
        if not hasattr(model, "n_jobs"):
            model.n_jobs = None
    return model


model = ensure_logreg_defaults(load_artifact(MODEL_PATH))
vectorizer = load_artifact(VECTORIZER_PATH)

st.title("Flipkart Review Sentiment Analysis")
st.write("Enter a product review to analyze sentiment.")

if model is None or vectorizer is None:
    st.warning(
        "Model or vectorizer not found. Train the model via `01_eda.ipynb` "
        "before running the app."
    )

review = st.text_area("Review Text", height=200, placeholder="Type or paste a review...")

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if review.strip() == "":
        st.warning("Please enter a review.")
    elif model is None or vectorizer is None:
        st.error("Missing artifacts. Please train and save the model first.")
    else:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0][pred]

        sentiment = "Positive" if pred == 1 else "Negative"
        if pred == 1:
            st.success(sentiment)
        else:
            st.error(sentiment)

        if proba is not None:
            st.write(f"Confidence: {proba:.2f}")

st.caption(
    "TF-IDF + Logistic Regression"
)

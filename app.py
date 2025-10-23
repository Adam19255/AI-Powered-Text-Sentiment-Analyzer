import streamlit as st
import joblib
import numpy as np
from transformers import pipeline

# ==================================
# 1. Load Classical Model
# ==================================
model_data = joblib.load("models/sentiment_model.joblib")
vectorizer = model_data["vectorizer"]
model = model_data["model"]

# ==================================
# 2. Load Transformer Model
# ==================================
@st.cache_resource # prevents reloading DistilBERT every time the page reruns
def load_transformer():
    return pipeline("sentiment-analysis", model="models/distilbert-imdb", tokenizer="models/distilbert-imdb")

transformer_model = load_transformer()

# ==================================
# 3. Streamlit UI
# ==================================
st.title("AI Sentiment Analyzer")

st.write("This demo compares a classical ML model (TF-IDF + Logistic Regression) with a Transformer model (DistilBERT).")

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # ========= Classical Prediction =========
        X_input = vectorizer.transform([user_input]) # Convert raw text → TF-IDF vector
        pred_class = model.predict(X_input)[0] # Predict label 0(neg) or 1(pos)
        pred_proba = model.predict_proba(X_input)[0][pred_class] # Get confidence score
        classical_label = "Positive" if pred_class == 1 else "Negative"
        classical_color = "green" if pred_class == 1 else "red"

        # ========= Transformer Prediction =========
        transformer_result = transformer_model(user_input)[0]
        raw_label = transformer_result['label']   # POSITIVE or NEGATIVE
        raw_score = transformer_result['score']   # probability

        # Determine NEUTRAL using threshold
        if raw_score < 0.35:
            transformer_label = "NEUTRAL"
            transformer_color = "blue"
            final_score = raw_score
        elif raw_score > 0.65:
            transformer_label = raw_label   # POSITIVE or NEGATIVE
            transformer_color = "green" if raw_label == "POSITIVE" else "red"
            final_score = raw_score
        else:
            transformer_label = "NEUTRAL"
            transformer_color = "blue"
            final_score = raw_score

        # ========= Display Classical =========
        st.subheader("Classical Model (TF-IDF + Logistic Regression)")
        st.markdown(
            f"<h3 style='color:{classical_color};'>Sentiment: {classical_label} ({pred_proba*100:.2f}%)</h3>",
            unsafe_allow_html=True
        )

        # ========= Display Transformer =========
        st.subheader("Transformer Model (DistilBERT)")
        st.markdown(
            f"<h3 style='color:{transformer_color};'>Sentiment: {transformer_label} ({final_score*100:.2f}%)</h3>",
            unsafe_allow_html=True
        )

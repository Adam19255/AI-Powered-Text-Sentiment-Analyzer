import streamlit as st
import joblib
import numpy as np

# ==============================
# 1. Load the saved model
# ==============================
# We load the dictionary we saved earlier (vectorizer + model)
model_data = joblib.load("models/sentiment_model.joblib")
vectorizer = model_data["vectorizer"]
model = model_data["model"]

# ==============================
# 2. Streamlit UI Setup
# ==============================
st.title("AI Sentiment Analyzer")

st.write("Type a movie review below and the AI will classify it as **Positive** or **Negative** sentiment.")

# Multiline text box for user input
user_input = st.text_area("Enter your text here:", height=200)

# ==============================
# 3. When user clicks Analyze
# ==============================
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Convert the input text to tf-idf vector
        X_input = vectorizer.transform([user_input])

        # Predict class (0 or 1) and probability
        pred_class = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0][pred_class]

        # Convert class number to label
        sentiment_label = "Positive" if pred_class == 1 else "Negative"

        # Choose color
        color = "green" if pred_class == 1 else "red"

        # ==============================
        # 4. Display predictions
        # ==============================
        st.markdown(
            f"<h2 style='color:{color};'>Sentiment: {sentiment_label}</h2>",
            unsafe_allow_html=True
        )
        st.write(f"**Confidence:** {pred_proba*100:.2f}%")

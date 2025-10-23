# 💬 AI-Powered Text Sentiment Analyzer

This project is an **end-to-end machine learning application** that analyzes the sentiment of text input using both a **classical ML model** and a **Transformer-based model (DistilBERT)**.

It was designed as a hands-on learning project to explore:

- Dataset handling and preprocessing
- Classical NLP modeling (TF-IDF + Logistic Regression)
- Transformer fine-tuning (DistilBERT)
- Model evaluation using Accuracy and F1-score
- Web deployment with Streamlit and Hugging Face Spaces

---

## 🚀 Demo

👉 **Live App:** [Hugging Face Space](https://huggingface.co/spaces/AdamShay/ai-sentiment-analyzer)

Enter any sentence to get predictions from:

- A **classical model** (TF-IDF + Logistic Regression)
- A **Transformer model** (DistilBERT fine-tuned on IMDB reviews)

---

## 🧠 Features

✅ Sentiment prediction (Positive / Negative / Neutral)  
✅ Confidence visualization bar  
✅ Real-time analysis through Streamlit UI  
✅ Local fine-tuned Transformer model for accuracy  
✅ Deployed publicly via Hugging Face Spaces

---

## 🏗️ Project Structure

AI-Powered-Text-Sentiment-Analyzer/
│
├── app.py # Streamlit web app
├── train_baseline.py # Classical ML training script
├── train_transformer.py # Transformer fine-tuning script
├── inspect_data.py # Dataset loader utility
├── requirements.txt # Python dependencies
├── README.md # This file
└── models/
├── sentiment_model.joblib # TF-IDF + Logistic Regression model
└── distilbert-imdb/ # Fine-tuned DistilBERT model files

## 📊 Dataset

Dataset used: [IMDB Movie Reviews Dataset](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews)

---

## 🧭 How to Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Powered-Text-Sentiment-Analyzer.git
cd AI-Powered-Text-Sentiment-Analyzer
pip install -r requirements.txt
streamlit run app.py

Then open http://localhost:8501 in your browser.
```

import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
import joblib

# We import the dataset loader you already created
from inspect_data import load_imdb_dataset

# =========================
# 1. LOAD THE DATA
# =========================
# This function reads the IMDB folder structure and returns
# two tuples: (train_texts, train_labels), (test_texts, test_labels)
# We shuffle the data to insure not all labels are the same
(train_texts, train_labels), (test_texts, test_labels) = load_imdb_dataset()
train_texts, train_labels = shuffle(train_texts, train_labels, random_state=42)

# =========================
# 2. USE ONLY A SMALL SUBSET (5000)
# =========================
# We are using B+YES as agreed
SUBSET_SIZE = 5000
train_texts = train_texts[:SUBSET_SIZE]
train_labels = train_labels[:SUBSET_SIZE]

# =========================
# 3. SPLIT TRAIN -> (train + validation)
# =========================
# test_size=0.20 means 20% will be used for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_texts,
    train_labels,
    test_size=0.20,
    random_state=42,  # ensures consistent results each run
    stratify=train_labels  # keeps positive/negative balance
)

# =========================
# 4. CONVERT TEXT TO NUMBERS (TF-IDF)
# =========================
# This step is critical: it changes raw text into numerical vectors
vectorizer = TfidfVectorizer(
    max_features=20000,   # limit vocabulary size for speed
    ngram_range=(1, 2)    # use single words and 2-word combinations
)

# The vectorizer "learns" vocabulary only from training text
X_train_tfidf = vectorizer.fit_transform(X_train)

# For validation/test we only "transform" - not "fit"
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test_texts)

# =========================
# 5. CHOOSE & TRAIN MODEL
# =========================
# Logistic Regression is a simple but effective classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# =========================
# 6. VALIDATION PERFORMANCE
# =========================
val_preds = clf.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, val_preds)
val_f1 = f1_score(y_val, val_preds)

print("=== Validation Results ===")
print("Accuracy:", round(val_accuracy, 4))
print("F1 Score:", round(val_f1, 4))

# =========================
# 7. TEST PERFORMANCE (FINAL EXAM)
# =========================
# Evaluate on the test set (unseen data)
test_preds = clf.predict(X_test_tfidf)
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)

print("=== Test Results ===")
print("Accuracy:", round(test_accuracy, 4))
print("F1 Score:", round(test_f1, 4))

# =========================
# 8. SAVE THE MODEL
# =========================
# We save both the vectorizer and classifier so we can load them later in our UI
os.makedirs("models", exist_ok=True)
joblib.dump({"vectorizer": vectorizer, "model": clf}, "models/sentiment_model.joblib")

print("\nModel saved to models/sentiment_model.joblib")

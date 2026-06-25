import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv")

# Fix columns (SMS Spam dataset format)
df = df.rename(columns={"v1": "label", "v2": "text"})

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Features & labels
X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=3000
)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")

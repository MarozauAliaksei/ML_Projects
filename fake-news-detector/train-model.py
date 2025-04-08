import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Label the data
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
combined = pd.concat([fake, true])
combined = shuffle(combined).reset_index(drop=True)

# Check class balance
print("Class distribution:\n", combined["label"].value_counts())

# Feature and label selection
X_text = combined["text"]
y = combined["label"]

# TF-IDF vectorization (adjusted)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X_text)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model with class_weight
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {acc * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully.")
# Import required libraries
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
true_news = pd.read_csv("dataset/True.csv")

print(fake_news.head())
print(true_news.head())
print("Fake News Shape:", fake_news.shape)
print("True News Shape:", true_news.shape)

print(fake_news.columns)
print(true_news.columns)
# Add labels
fake_news["label"] = 0
true_news["label"] = 1

print(fake_news.head())
print(true_news.head())
# Merge fake and true news datasets
news = pd.concat([fake_news, true_news], ignore_index=True)

print(news.head())
print("Total Rows:", news.shape)
# Shuffle the dataset
news = news.sample(frac=1).reset_index(drop=True)

print(news.head())
# Check for missing values
print(news.isnull().sum())
# Keep only text and label
news = news[["text", "label"]]
# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
stop_words = set(stopwords.words("english"))
news["text"] = news["text"].apply(clean_text)

news["text"] = news["text"].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

print(news.head())

print(news.head())

print(news.head())
print(news.columns)
# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = vectorizer.fit_transform(news["text"])
y = news["label"]

print(X.shape)
print(y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
# Train Logistic Regression model
model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print(classification_report(y_test, y_pred))
# Save the trained model
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and Vectorizer saved successfully!")
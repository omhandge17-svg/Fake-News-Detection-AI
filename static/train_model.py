# Import required libraries
import pandas as pd
import numpy as np
# Load datasets
fake_news = pd.read_csv("dataset/Fake.csv")
true_news = pd.read_csv("dataset/True.csv")

print(fake_news.head())
print(true_news.head())
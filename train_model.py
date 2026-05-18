import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pickle

# Load dataset
iris = load_iris()

X = iris.data

# Train KMeans model
model = KMeans(n_clusters=3, random_state=42)

model.fit(X)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
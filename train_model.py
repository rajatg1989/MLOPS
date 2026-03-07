from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

data = load_iris()

X = data.data
y = data.target

model = RandomForestClassifier()

model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")
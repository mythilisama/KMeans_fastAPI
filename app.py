from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)

model.fit(X, y)

# Create app
app = FastAPI()

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction route
@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    flower = iris.target_names[prediction]

    return {
        "prediction": flower
    }
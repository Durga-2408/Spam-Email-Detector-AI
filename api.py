from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.post("/predict")
def predict(email: str):
    vec = vectorizer.transform([email])
    result = model.predict(vec)
    return {"prediction": int(result[0])}
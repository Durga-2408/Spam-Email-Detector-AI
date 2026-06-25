from turtle import st

from fastapi import FastAPI
import joblib


st.title("📧 Spam Email Detector")
st.write("Enter an email message to check whether it is Spam or Not Spam.")

app = FastAPI()

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.post("/predict")
def predict(email: str):
    vec = vectorizer.transform([email])
    result = model.predict(vec)
    probability = model.predict_proba(vec)
    return {"prediction": int(result[0]), "probability": probability[0].tolist()}
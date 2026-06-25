import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam Email Detector")

email_text = st.text_area("Enter Email Text")

if st.button("Check"):
    data = vectorizer.transform([email_text])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Spam Email")
    else:
        st.success("Not Spam Email")
import streamlit as st
import joblib

# Load model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="Spam Detector AI", page_icon="📧")

# Title
st.title("📧 Spam Email Detector AI")
st.write("Check whether your email is Spam or Not using AI")

# Input
input_text = st.text_area("Enter Email Text")

# Button
if st.button("Check Email"):

    if input_text.strip():

        # Transform text
        vec = vectorizer.transform([input_text])

        # Prediction
        result = model.predict(vec)

        # Confidence
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        # Output
        if result[0] == 1:
            st.error("🚨 Spam Email Detected!")
        else:
            st.success("✅ This is a Normal Email")

        st.info(f"Confidence Score: {confidence:.2f}%")

    else:
        st.warning("Please enter email text first!")

# Footer
st.markdown("---")
st.caption("Built with Streamlit + Machine Learning")
#Confidence score
vec = vectorizer.transform([input_text])
prob = model.predict_proba(vec)[0]
st.write("Spam Probability:", f"{prob[1]*100:.2f}%")
st.write("Normal Probability:", f"{prob[0]*100:.2f}%")
#why spam
if "http" in input_text:
    st.warning("Reason: Contains suspicious link")
if "$" in input_text or "free" in input_text:
    st.warning("Reason: Scam keywords detected")
#input validation
if input_text.stip()== "":
    st.warning("Please enter email text")
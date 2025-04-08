import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("""
This app uses a machine learning model to classify whether a piece of news is **real** or **fake**.
""")

# User input
user_input = st.text_area("Enter a news article or paragraph below:", height=250)

# Prediction
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        probability = model.predict_proba(input_vector)[0]

        if prediction[0] == 1:
            st.success("🟢 This appears to be REAL news.")
        else:
            st.error("🔴 This appears to be FAKE news.")

        st.markdown(f"**Confidence:** Real {probability[1]*100:.2f}% | Fake {probability[0]*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Aliaksei Marozau")

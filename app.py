import streamlit as st
from transformers import pipeline

# settings
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 AI Customer Sentiment Analyzer")
st.write("Write any review, feedback, or sentence below, and the AI will analyze its sentiment (Positive or Negative) in real-time!")

st.write("---")

# Load model (use Cache for faster)
@st.cache_resource
def load_model():
    # Load default sentiment analysis model from Hugging Face
    return pipeline("sentiment-analysis")

analyzer = load_model()

# user input
user_input = st.text_area("Enter your text here:", "I absolutely love this product! The quality is amazing and the delivery was incredibly fast.")

# button and prediction
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("AI is analyzing the text... Please wait..."):
            # model is predicting
            result = analyzer(user_input)[0]
            sentiment = result['label']
            confidence = result['score'] * 100

            st.write("### 📊 Analysis Result:")
            
            # shoe message and color according to result
            if sentiment == 'POSITIVE':
                st.success(f"🌟 **Sentiment:** {sentiment} \n\n🎯 **Confidence Score:** {confidence:.2f}%")
            elif sentiment == 'NEGATIVE':
                st.error(f"⚠️ **Sentiment:** {sentiment} \n\n🎯 **Confidence Score:** {confidence:.2f}%")
            else:
                st.info(f"😐 **Sentiment:** {sentiment} \n\n🎯 **Confidence Score:** {confidence:.2f}%")

st.write("---")
st.caption("Powered by Hugging Face Transformers & Streamlit | Built for NLP Portfolio")

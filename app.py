import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ওয়েবসাইটের মূল সেটিং
st.set_page_config(page_title="Advanced AI Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 Advanced Customer Feedback Analyzer")
st.write("Analyze Sentiments, Detect Emotions, Generate Word Clouds, and Export Reports in seconds!")
st.write("---")

# মডেল লোড করা
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# ট্যাব তৈরি করা
tab1, tab2 = st.tabs(["💬 Single Text & Emotion", "📂 Bulk CSV Analytics Dashboard"])

# ==========================================
# TAB 1: Single Text Analysis (আগের মতোই আছে)
# ==========================================
with tab1:
    st.subheader("Analyze a Single Review")
    analysis_type = st.radio("Choose Analysis Type:", ["Multilingual Sentiment (Positive/Negative/Neutral)", "Emotion Detection (English only)"])
    user_input = st.text_area("Enter text here:", "I absolutely love this product!")

    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("AI is thinking..."):
                if "Sentiment" in analysis_type:
                    analyzer = load_sentiment_model()
                    result = analyzer(user_input)[0]
                    st.success(f"**Sentiment:** {result['label'].upper()} (Confidence: {result['score'] * 100:.2f}%)")
                else:
                    emotion_analyzer = load_emotion_model()
                    result = emotion_analyzer(user_input)[0]
                    st.info(f"**Detected Emotion:** {result['label'].capitalize()} (Confidence: {result['score'] * 100:.2f}%)")

# ==========================================
# TAB 2: Bulk CSV Analytics Dashboard (নতুন ফিচার যুক্ত)
# ==========================================
with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())
        
        text_column = st.selectbox("Select the column containing text/reviews:", df.columns)
        
        if st.button("Generate Analytics Dashboard"):
            with st.spinner("Analyzing bulk data... Please wait."):
                analyzer = load_sentiment_model()
                data_to_analyze = df[text_column].dropna().astype(str).head(100).tolist()
                
                # প্রেডিকশন
                results = analyzer(data_to_analyze)
                sentiments = [res['label'] for res in results]
                
                df_result = pd.DataFrame({
                    "Review": data_to_analyze,
                    "Sentiment": sentiments
                })
                
                st.success("✅ Analysis Complete!")
                
                # --- নতুন ফিচার ১: Download Report ---
                st.write("### 📄 Analysis Report")
                st.dataframe(df_result)
                
                # CSV ফাইলে কনভার্ট করা
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                
                # ডাউনলোড বাটন
                st.download_button(
                    label="📥 Download Analyzed Data (CSV)",
                    data=csv_data,
                    file_name="AI_Sentiment_Report.csv",
                    mime="text/csv"
                )
                
                st.write("---")
                
                # ড্যাশবোর্ড গ্রাফ (পাই চার্ট)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 📊 Sentiment Distribution")
                    sentiment_counts = df_result['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                                 color='Sentiment',
                                 color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- নতুন ফিচার ২: Word Cloud ---
                with col2:
                    st.write("### ☁️ Word Cloud (Most Used Words)")
                    # সব রিভিউ একসাথে যুক্ত করা
                    all_words = " ".join(review for review in df_result['Review'])
                    
                    # ওয়ার্ড ক্লাউড তৈরি করা
                    wordcloud = WordCloud(width=800, height=600, background_color='white', colormap='viridis').generate(all_words)
                    
                    # ছবি হিসেবে দেখানো
                    fig_wc, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off") # গ্রাফের চারপাশের বর্ডার বাদ দেওয়া
                    st.pyplot(fig_wc)

st.write("---")
st.caption("Powered by Hugging Face & Streamlit | Enterprise NLP Solution")

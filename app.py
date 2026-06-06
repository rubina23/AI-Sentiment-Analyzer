import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from youtube_comment_downloader import *
import itertools

# ওয়েবসাইটের মূল সেটিং
st.set_page_config(page_title="Ultimate AI Analyzer", page_icon="🚀", layout="wide")

st.title("🚀 Ultimate Customer Feedback & YouTube Analyzer")
st.write("Analyze Sentiments, Detect Emotions, Generate Auto-Replies, and Scrape Live YouTube Comments!")
st.write("---")

# ==========================================
# AI মডেল লোড করা (Cache ব্যবহার করে)
# ==========================================
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_reply_model():
    # অটো-রিপ্লাইয়ের জন্য গুগলের টেক্সট জেনারেশন মডেল
    return pipeline("text2text-generation", model="google/flan-t5-small")

# ৪টি ট্যাব তৈরি করা
tab1, tab2, tab3, tab4 = st.tabs(["💬 Single Text", "📂 Bulk CSV", "🤖 Auto-Reply Generator", "▶️ Live YouTube Analysis"])

# ==========================================
# TAB 1: Single Text Analysis
# ==========================================
with tab1:
    st.subheader("Analyze a Single Review")
    analysis_type = st.radio("Choose Analysis Type:", ["Multilingual Sentiment", "Emotion Detection (English)"])
    user_input = st.text_area("Enter text here:", "I absolutely love this product!")

    if st.button("Analyze Text"):
        if user_input.strip() != "":
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
# TAB 2: Bulk CSV Analytics Dashboard
# ==========================================
with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_column = st.selectbox("Select the text column:", df.columns)
        
        if st.button("Generate Dashboard"):
            with st.spinner("Analyzing bulk data..."):
                analyzer = load_sentiment_model()
                data_to_analyze = df[text_column].dropna().astype(str).head(100).tolist()
                results = analyzer(data_to_analyze)
                sentiments = [res['label'] for res in results]
                
                df_result = pd.DataFrame({"Review": data_to_analyze, "Sentiment": sentiments})
                st.success("✅ Analysis Complete!")
                st.dataframe(df_result.head(10))
                
                # Download Button
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Analyzed Data", data=csv_data, file_name="CSV_Report.csv", mime="text/csv")
                st.write("---")
                
                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_counts = df_result['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title="Sentiment Distribution",
                                 color='Sentiment', color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### ☁️ Word Cloud")
                    all_words = " ".join(review for review in df_result['Review'])
                    wordcloud = WordCloud(width=800, height=600, background_color='white', colormap='viridis').generate(all_words)
                    fig_wc, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)

# ==========================================
# TAB 3: Auto-Reply Generator
# ==========================================
with tab3:
    st.subheader("Generate Professional AI Replies")
    st.write("Paste a customer review, and AI will generate a polite and professional response.")
    
    review_input = st.text_area("Customer Review:", "My package arrived completely broken. I want a refund immediately!")
    
    if st.button("Generate Reply"):
        if review_input.strip() != "":
            with st.spinner("Drafting response..."):
                reply_model = load_reply_model()
                # AI কে প্রম্পট বা নির্দেশ দেওয়া হচ্ছে
                prompt = f"Write a professional and polite customer service reply to this review: '{review_input}'"
                
                # ম্যাক্সিমাম লেন্থ একটু বাড়িয়ে দেওয়া হয়েছে সুন্দর উত্তরের জন্য
                generated_text = reply_model(prompt, max_length=100, do_sample=True)[0]['generated_text']
                
                st.write("### ✍️ AI Suggested Reply:")
                st.success(generated_text)

# ==========================================
# TAB 4: Live YouTube Analysis
# ==========================================
with tab4:
    st.subheader("Live YouTube Video Comments Analyzer")
    st.write("Paste any YouTube video link to instantly scrape and analyze the top comments!")
    
    yt_url = st.text_input("Enter YouTube Video URL:", "https://www.youtube.com/watch?v=1-68pFs_HIA")
    
    if st.button("Scrape & Analyze Comments"):
        if yt_url != "":
            with st.spinner("Downloading comments from YouTube... This may take a few seconds."):
                try:
                    downloader = YoutubeCommentDownloader()
                    # ইউটিউব থেকে প্রথম 50 টি কমেন্ট টেনে আনা হচ্ছে
                    comments_generator = downloader.get_comments_from_url(yt_url, sort_by=SORT_BY_POPULAR)
                    comments_list = list(itertools.islice(comments_generator, 50))
                    
                    if len(comments_list) > 0:
                        comment_texts = [comment['text'] for comment in comments_list]
                        
                        st.write("Analyzing Sentiments...")
                        analyzer = load_sentiment_model()
                        results = analyzer(comment_texts)
                        sentiments = [res['label'] for res in results]
                        
                        df_yt = pd.DataFrame({"Comment": comment_texts, "Sentiment": sentiments})
                        
                        st.success(f"✅ Successfully analyzed {len(df_yt)} comments!")
                        
                        # ইউটিউব ড্যাশবোর্ড
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment_counts = df_yt['Sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title="YouTube Sentiment Overview",
                                         color='Sentiment', color_discrete_map={'positive':'#2ecc71', 'negative':'#e74c3c', 'neutral':'#95a5a6'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.write("### 📄 Latest Comments Data")
                            st.dataframe(df_yt)
                            
                        # Download YouTube Data
                        csv_yt = df_yt.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download YouTube Comments", data=csv_yt, file_name="YouTube_Analysis.csv", mime="text/csv")
                        
                    else:
                        st.warning("No comments found or comments are disabled for this video.")
                        
                except Exception as e:
                    st.error("Could not fetch comments. Please ensure the URL is correct and the video is public.")

st.write("---")
st.caption("Powered by Hugging Face, YouTube Scraper & Streamlit | Ultimate NLP Pipeline")

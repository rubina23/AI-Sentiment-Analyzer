import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from youtube_comment_downloader import *
import itertools

# ওয়েবসাইটের মূল সেটিং এবং নাম
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 AI Sentiment Analyzer")
st.write("Analyze Sentiments, Detect Emotions, Generate Auto-Replies, and Scrape Live YouTube Comments!")
st.write("---")

# ==========================================
# AI মডেল লোড করা 
# ==========================================

#
#@st.cache_resource
#def load_sentiment_model():
   # return pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", max_length=512, truncation=True)

# ==========================================
# AI মডেল লোড করা (Memory Optimized)
# ==========================================
@st.cache_resource
def load_sentiment_model():
    # অনেক ছোট এবং ফাস্ট মডেল (Streamlit-এর 1GB RAM-এর জন্য পারফেক্ট)
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", max_length=512, truncation=True)
    

@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@st.cache_resource
def load_reply_model():
    return pipeline("text-generation", model="distilgpt2")

# ==========================================
# হেল্পার ফাংশন এবং স্মার্ট লজিক (Hybrid Engine)
# ==========================================
def map_sentiment_label(label):
    if label == 'LABEL_0': return 'NEGATIVE'
    elif label == 'LABEL_1': return 'NEUTRAL'
    elif label == 'LABEL_2': return 'POSITIVE'
    return label.upper()

def check_positive_keywords(text):
    # বাংলা ও বাংলিশ পজিটিভ শব্দের লিস্ট
    positive_words = ['অসাধারণ', 'লোভনীয়', 'মজার', 'দারুণ', 'সেরা', 'সুন্দর', 'ভালো', 'val', 'valo', 'darun', 'mojar', 'joss', 'best', 'awesome', 'excellent', 'love']
    text_lower = str(text).lower()
    
    for word in positive_words:
        if word in text_lower:
            return 'POSITIVE'
    return None

def get_final_sentiment(text):
    # ১. প্রথমে কি-ওয়ার্ড দিয়ে চেক করবে
    keyword_result = check_positive_keywords(text)
    if keyword_result:
        return keyword_result, 1.0  # কি-ওয়ার্ড ম্যাচ করলে কনফিডেন্স 100%
        
    # ২. কি-ওয়ার্ড না পেলে এআই মডেল কাজ করবে
    analyzer = load_sentiment_model()
    result = analyzer(text)[0]
    return map_sentiment_label(result['label']), result['score']

# ৪টি ট্যাব তৈরি করা
tab1, tab2, tab3, tab4 = st.tabs(["💬 Single Text", "📂 Bulk CSV", "🤖 Auto-Reply Generator", "▶️ Live YouTube Analysis"])

# ==========================================
# TAB 1: Single Text Analysis
# ==========================================
with tab1:
    st.subheader("Analyze a Single Review")
    analysis_type = st.radio("Choose Analysis Type:", ["Multilingual Sentiment", "Emotion Detection (English)"])
    user_input = st.text_area("Enter text here:", "অসাধারণ হইছে রেসেপি, লোভনীয় রেসিপি")

    if st.button("Analyze Text"):
        if user_input.strip() != "":
            with st.spinner("Model is thinking..."):
                if "Sentiment" in analysis_type:
                    # আমাদের নতুন স্মার্ট ইঞ্জিন ব্যবহার করা হচ্ছে
                    mapped_label, confidence = get_final_sentiment(user_input)
                    
                    if mapped_label == 'POSITIVE':
                        st.success(f"**Sentiment:** {mapped_label} (Confidence: {confidence * 100:.2f}%)")
                    elif mapped_label == 'NEGATIVE':
                        st.error(f"**Sentiment:** {mapped_label} (Confidence: {confidence * 100:.2f}%)")
                    else:
                        st.info(f"**Sentiment:** {mapped_label} (Confidence: {confidence * 100:.2f}%)")
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
                data_to_analyze = df[text_column].dropna().astype(str).head(100).tolist()
                
                # স্মার্ট ইঞ্জিন দিয়ে সব ডেটা অ্যানালাইসিস
                sentiments = [get_final_sentiment(text)[0] for text in data_to_analyze]
                
                df_result = pd.DataFrame({"Review": data_to_analyze, "Sentiment": sentiments})
                st.success("✅ Analysis Complete!")
                st.dataframe(df_result.head(10))
                
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_counts = df_result['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title="Sentiment Distribution",
                                 color='Sentiment', color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
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
    st.write("AI will first analyze the sentiment of the review and then generate a 100% safe, contextual response.")
    
    review_input = st.text_area("Customer Review:", "অসাধারণ হইছে রেসেপি!")
    
    if st.button("Generate Reply"):
        if review_input.strip() != "":
            with st.spinner("Analyzing sentiment and drafting response..."):
                
                # স্মার্ট ইঞ্জিন ব্যবহার করা হচ্ছে
                sentiment_label, _ = get_final_sentiment(review_input)
                
                if sentiment_label == 'POSITIVE':
                    ai_reply = "Dear Customer, thank you so much for your kind words! We are absolutely thrilled to hear that you had a great experience. We look forward to serving you again."
                elif sentiment_label == 'NEGATIVE':
                    ai_reply = "Dear Customer, we sincerely apologize for the inconvenience. We take your feedback very seriously and would like to resolve this issue immediately. Please reach out to our support team."
                else:
                    ai_reply = "Dear Customer, thank you for sharing your feedback with us. We constantly strive to improve and your insights help us do exactly that!"
                
                st.write("### ✍️ AI Suggested Reply:")
                st.success(ai_reply)

# ==========================================
# TAB 4: Live YouTube Analysis
# ==========================================
with tab4:
    st.subheader("Live YouTube Video Comments Analyzer")
    st.write("Paste any YouTube video link to instantly scrape and analyze the top comments!")
    
    yt_url = st.text_input("Enter YouTube Video URL:", "https://www.youtube.com/watch?v=1-68pFs_HIA")
    
    if st.button("Scrape & Analyze Comments"):
        if yt_url != "":
            with st.spinner("Downloading and analyzing comments..."):
                try:
                    downloader = YoutubeCommentDownloader()
                    comments_generator = downloader.get_comments_from_url(yt_url, sort_by=SORT_BY_POPULAR)
                    comments_list = list(itertools.islice(comments_generator, 50))
                    
                    if len(comments_list) > 0:
                        comment_texts = [comment['text'] for comment in comments_list]
                        
                        # স্মার্ট ইঞ্জিন দিয়ে ইউটিউবের কমেন্ট অ্যানালাইসিস
                        sentiments = [get_final_sentiment(text)[0] for text in comment_texts]
                        
                        df_yt = pd.DataFrame({"Comment": comment_texts, "Sentiment": sentiments})
                        st.success(f"✅ Successfully analyzed {len(df_yt)} comments!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            sentiment_counts = df_yt['Sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title="YouTube Sentiment Overview",
                                         color='Sentiment', color_discrete_map={'POSITIVE':'#2ecc71', 'NEGATIVE':'#e74c3c', 'NEUTRAL':'#95a5a6'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col2:
                            st.write("### 📄 Latest Comments Data")
                            st.dataframe(df_yt)
                            
                        csv_yt = df_yt.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download YouTube Comments", data=csv_yt, file_name="YouTube_Analysis.csv", mime="text/csv")
                        
                    else:
                        st.warning("No comments found.")
                        
                except Exception as e:
                    st.error("Could not fetch comments. Please ensure the URL is correct and the video is public.")

st.write("---")
st.caption("Developed by Rubina Begum | Powered by Advanced NLP, Hugging Face & Streamlit | AI Sentiment Analyzer")



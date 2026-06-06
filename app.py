import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# settings
st.set_page_config(page_title="Advanced AI Sentiment Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Advanced AI Sentiment & Emotion Analyzer")
st.write("Analyze customer sentiments in multiple languages (English, Bengali, etc.), detect specific emotions, and process bulk reviews via CSV!")

st.write("---")

# Load two model using Cache
@st.cache_resource
def load_models():
    # multitingual sentiment model (bangla, English,...)
    sentiment_model = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    # emotion model
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return sentiment_model, emotion_model

# load model
with st.spinner("Loading AI Models... This might take a few seconds on the first run."):
    sentiment_analyzer, emotion_analyzer = load_models()

# separate in 2 tabs
tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📁 Bulk CSV Analysis (Dashboard)"])

# -------------------------------------------------------------------------
# Tab 1: Single text analysis (Multilingual + Emotion)
# -------------------------------------------------------------------------
with tab1:
    st.header("Analyze a Single Review")
    user_input = st.text_area("Enter text here (You can type in English, Bengali, Hindi, etc.):", 
                              "The product is absolutely amazing! I love it, but the delivery was a bit late.")
    
    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("AI is thinking..."):
                # prediction
                sentiment_result = sentiment_analyzer(user_input)[0]
                emotion_result = emotion_analyzer(user_input)[0]
                
                # result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("💡 Sentiment (Multilingual)")
                    sentiment = sentiment_result['label'].upper()
                    conf_s = sentiment_result['score'] * 100
                    if "POSITIVE" in sentiment:
                        st.success(f"**{sentiment}** ({conf_s:.1f}%)")
                    elif "NEGATIVE" in sentiment:
                        st.error(f"**{sentiment}** ({conf_s:.1f}%)")
                    else:
                        st.info(f"**{sentiment}** ({conf_s:.1f}%)")
                
                with col2:
                    st.subheader("🎭 Emotion Detected")
                    emotion = emotion_result['label'].capitalize()
                    conf_e = emotion_result['score'] * 100
                    st.info(f"**Emotion:** {emotion} 🧐 \n\n**Confidence:** {conf_e:.1f}%")

# -------------------------------------------------------------------------
# Tab 2: Upload Dashboard & CSV 
# -------------------------------------------------------------------------
with tab2:
    st.header("Bulk Review Analysis Dashboard")
    st.write("Upload a CSV file containing a column named **'review'** or **'text'** to analyze hundreds of feedbacks at once.")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # read DB
        df = pd.read_csv(uploaded_file)
        
        # select column by user (work any excel file)
        text_column = st.selectbox("Select the column containing the text/reviews:", df.columns)
        
        if st.button("Generate Dashboard"):
            with st.spinner("AI is analyzing all reviews... Please wait."):
                
                # First 100 reviews (যাতে ফ্রি সার্ভার ক্র্যাশ না করে)
                texts = df[text_column].dropna().astype(str).tolist()[:100] 
                
                # Analysis all review
                results = sentiment_analyzer(texts)
                sentiments = [res['label'] for res in results]
                
                # add new col
                df_result = pd.DataFrame({"Review": texts, "Sentiment": sentiments})
                
                # ড্যাDashboard column
                colA, colB = st.columns([1, 1])
                
                with colA:
                    st.subheader("📊 Sentiment Distribution")
                    # Create Pie chart using Plotly
                    sentiment_counts = df_result['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                                 color='Sentiment', 
                                 color_discrete_map={'positive':'#00cc96', 'negative':'#EF553B', 'neutral':'#636EFA'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with colB:
                    st.subheader("📄 Processed Data")
                    st.dataframe(df_result, height=350)
                    
                st.success("✅ Analysis Complete! Showing results for up to 100 records.")

st.write("---")
st.caption("Powered by Hugging Face Transformers & Streamlit | Advanced NLP Portfolio")

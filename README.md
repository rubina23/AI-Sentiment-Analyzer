# 🧠 AI Sentiment Analyzer & NLP Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-F9AB00.svg)

This is an AI-powered Natural Language Processing (NLP) dashboard built with **Streamlit** and **Hugging Face**. This tool helps businesses analyze customer feedback, detect emotions, generate context-aware automated replies, and scrape live YouTube comments for real-time market sentiment analysis.

🔗 **Live Demo:** [https://ai-sentiment-analyzer-3q.streamlit.app/]

---

## 🚀 Key Features

* **Single Text Analysis (Multilingual):** Analyze customer reviews in multiple languages. Includes a **Hybrid Rule-Based Engine** to override AI contextual bias for complex local languages (e.g., Bengali, Banglish).
* **Emotion Detection:** Accurately detects 7 standard human emotions (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral) from English text.
* **Bulk CSV Analytics:** Upload a CSV file of customer reviews and instantly generate a beautiful interactive dashboard containing **Pie Charts** (via Plotly) and **Word Clouds**.
* **Context-Aware Auto-Reply Generator:** Simulates enterprise customer support. It first detects the sentiment of the review and uses a generative AI model (`distilgpt2`) to craft a 100% safe, professional response.
* **Live YouTube Comment Scraper:** Paste any YouTube video link to instantly scrape the top comments, analyze their sentiments, visualize the data, and download the report as a CSV.

---

## 🛠️ Technology Stack

* **Frontend & Framework:** Streamlit
* **AI & Machine Learning:** Hugging Face `transformers` (Pipeline, AutoTokenizer)
* **Data Manipulation:** Pandas
* **Data Visualization:** Plotly Express, Matplotlib, WordCloud
* **Web Scraping:** `youtube-comment-downloader`

### 🧠 AI Models Used (Optimized for 1GB Cloud Environment):
1. **Sentiment Analysis:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
2. **Emotion Detection:** `j-hartmann/emotion-english-distilroberta-base`
3. **Text Generation:** `distilgpt2`

---

## ⚙️ How to Run Locally

Follow these steps to run the application on your local machine:

1. **Clone the repository:**
```bash
   git clone https://github.com/rubina23/AI-Sentiment-Analyzer
   cd your-repo-name

```
2. **Install the required dependencies:**
```
  pip install -r requirements.txt

```
3. **Run the Streamlit app:**
```
  streamlit run app.py

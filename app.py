import streamlit as st
import json
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import subprocess
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Streamlit app
st.title("TikTok Comments Sentiment Analysis")
st.write("Enter a TikTok video URL to analyze the sentiment of its comments.")

# Input URL
url = st.text_input("TikTok Video URL", placeholder="https://www.tiktok.com/@username/video/123456789")

# Function to run yt-dlp and fetch comments
def fetch_comments(url, temp_dir, retries=3):
    for attempt in range(retries):
        try:
            output_file = os.path.join(temp_dir, "tiktok_info.json")
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-comments",
                "--no-warnings",
                "--output",
                output_file,
                url
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            for f in os.listdir(temp_dir):
                if f.endswith(".info.json"):
                    return os.path.join(temp_dir, f)
            return None
        except subprocess.CalledProcessError as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                continue
            st.error(f"Error fetching comments: {e.stderr}")
            return None

# Function to analyze comments
def analyze_comments(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract comments (TikTok comments are nested under 'comments')
        comments = [
            {"text": c.get("text", ""), "likes": c.get("digg_count", 0)}
            for c in data.get("comments", [])
        ]
        if not comments:
            return None, None
        
        df = pd.DataFrame(comments)
        sid = SentimentIntensityAnalyzer()
        df["sentiment"] = df["text"].apply(lambda t: sid.polarity_scores(t)["compound"])
        df["bucket"] = pd.cut(df["sentiment"], [-1, -0.05, 0.05, 1], labels=["neg", "neu", "pos"])
        
        # Calculate sentiment distribution as percentages
        sentiment_dist = df["bucket"].value_counts(normalize=True) * 100
        sentiment_dist = sentiment_dist.round(2).to_dict()
        
        return df, sentiment_dist
    except Exception as e:
        st.error(f"Error analyzing comments: {str(e)}")
        return None, None

# Process when URL is provided
if url:
    with st.spinner("Fetching and analyzing comments..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = fetch_comments(url, temp_dir)
            if json_file:
                # Provide JSON download link
                with open(json_file, "rb") as f:
                    st.download_button(
                        label="Download JSON File",
                        data=f,
                        file_name="tiktok_info.json",
                        mime="application/json"
                    )
                
                # Analyze comments
                df, sentiment_dist = analyze_comments(json_file)
                if df is not None:
                    st.subheader("Analysis Results")
                    st.write(f"**Total Comments**: {len(df)}")
                    st.write("**Sentiment Distribution (%):**")
                    st.json(sentiment_dist)
                    
                    # Create pie chart
                    if sentiment_dist:
                        chart_data = pd.DataFrame(
                            list(sentiment_dist.items()),
                            columns=["Sentiment", "Percentage"]
                        ).set_index("Sentiment")
                        st.subheader("Sentiment Distribution Pie Chart")
                        st.pyplot(
                            chart_data.plot.pie(
                                y="Percentage",
                                labels=chart_data.index,
                                autopct="%1.1f%%",
                                figsize=(6, 6)
                            ).get_figure()
                        )
                else:
                    st.warning("No comments found or analysis failed.")
            else:
                st.error("Failed to fetch comments. Please check the URL or try again.")

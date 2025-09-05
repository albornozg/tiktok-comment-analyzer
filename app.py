import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
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

# Add headless mode toggle
use_headless = st.checkbox("Use Headless Mode", value=True, help="Disable for non-headless testing to bypass anti-bot measures.")

# Input URL
url = st.text_input("TikTok Video URL", placeholder="https://www.tiktok.com/@username/video/123456789")

# Function to scrape comments using Selenium
def fetch_comments(url, max_comments=100, retries=3):
    comments = []
    for attempt in range(retries):
        try:
            st.write(f"Attempt {attempt + 1}: Setting up Chrome options...")
            # Set up Chrome options
            chrome_options = Options()
            if use_headless:
                chrome_options.add_argument("--headless")
                st.write("Using headless mode.")
            else:
                st.write("Using non-headless mode.")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36")
            
            st.write("Initializing WebDriver...")
            driver = webdriver.Chrome(options=chrome_options)
            
            st.write("Navigating to URL...")
            driver.get(url)
            
            st.write("Waiting for comments to load...")
            WebDriverWait(driver, 50).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-e2e='comment-level-1']"))
            )
            
            st.write("Scrolling to load comments...")
            last_height = driver.execute_script("return document.body.scrollHeight")
            while len(comments) < max_comments:
                comment_elements = driver.find_elements(By.CSS_SELECTOR, "span[data-e2e='comment-level-1']")
                for elem in comment_elements:
                    text = elem.text.strip()
                    if text and len(comments) < max_comments:
                        comments.append({"text": text, "likes": 0})
                if len(comments) >= max_comments:
                    break
                
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            driver.quit()
            if comments:
                return comments
            else:
                st.warning(f"Attempt {attempt + 1} found no comments. Retrying...")
                continue
        except Exception as e:
            if 'driver' in locals():
                driver.quit()
            st.error(f"Attempt {attempt + 1} failed at: {str(e)}")
            if attempt < retries - 1:
                st.warning("Retrying...")
                time.sleep(2)
                continue
            st.error(f"Error fetching comments: {str(e)}")
            return None
    return None

# Function to analyze comments
def analyze_comments(comments):
    try:
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
        # Use temporary directory for JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            comments = fetch_comments(url)
            if comments:
                # Save comments as JSON for download
                json_file = os.path.join(temp_dir, "tiktok_comments.json")
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(comments, f, ensure_ascii=False)
                
                # Provide JSON download link
                with open(json_file, "rb") as f:
                    st.download_button(
                        label="Download JSON File",
                        data=f,
                        file_name="tiktok_comments.json",
                        mime="application/json"
                    )
                
                # Analyze comments
                df, sentiment_dist = analyze_comments(comments)
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







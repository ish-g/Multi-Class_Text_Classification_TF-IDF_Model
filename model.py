import streamlit as st
import nltk
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import pickle

# Load positive tweets
positive_tweets = twitter_samples.raw('positive_tweets.json')

# Load negative tweets
negative_tweets = twitter_samples.raw('negative_tweets.json')

# Load a general tweet file (can be a mix of positive and negative)
all_tweets = twitter_samples.raw('tweets.20150430-223406.json')

def load_json_lines(json_data):
    tweets = []
    for line in json_data.splitlines():
        if line.strip():  # Skip empty lines
            tweets.append(json.loads(line))
    return tweets

# Load and parse the tweets data
positive_tweets_list = load_json_lines(positive_tweets)
negative_tweets_list = load_json_lines(negative_tweets)

# Convert to DataFrames
positive_df = pd.DataFrame(positive_tweets_list)
negative_df = pd.DataFrame(negative_tweets_list)

# Add labels
positive_df['label'] = 'positive'
negative_df['label'] = 'negative'

# Combine into a single DataFrame
df = pd.concat([positive_df, negative_df], ignore_index=True)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and SVC
pipeline = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear', probability=True)
)

# Train the model
pipeline.fit(X_train, y_train)

with open('tweet_sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
import streamlit as st
import pickle

# Load the model from the pickle file
with open('tweet_sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title('Twitter Sentiment Analysis')

# Input for tweet
tweet = st.text_area('Enter a tweet to analyze:', '')

# Add an "Analyze" button
if st.button('Analyze'):
    if tweet:
        # Predict sentiment
        prediction = model.predict([tweet])[0]
        prob = model.predict_proba([tweet])[0]
        
        st.write(f'Sentiment: {prediction.capitalize()}')
        st.write(f'Probability: Positive: {prob[1]:.2f}, Negative: {prob[0]:.2f}')
    else:
        st.write("Please enter a tweet.")
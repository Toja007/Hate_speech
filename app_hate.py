import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re

# Load the trained model
model = pickle.load(open("model_hate.pkl", 'rb'))

# Load the TF-IDF vectorizer
cv = pickle.load(open("vectorizer_hate.pkl", 'rb'))

# Function to preprocess text
def preprocess_text(text):
    pattern = r'[^a-zA-Z0-9\s]'
    
    # Use re.sub to replace the matched pattern with an empty string
    text = re.sub(pattern, '', text)
    text = str.lower(text)
    return text

# Function to predict hate speech
def check(text):
    text = preprocess_text(text)
    text = cv.transform([text])
    pred = model.predict(text)
    return pred[0]

# Streamlit app
def main():
    st.title('Hate Speech Detection')
    st.write('Enter a tweet to classify it as hate speech, offensive speech, or non-hate speech.')

    # Text input for user to enter tweet
    tweet_text = st.text_area('Enter tweet text', '')

    # Button to predict
    if st.button('Predict'):
        if tweet_text:
            prediction = check(tweet_text)
            if prediction == 'Hate Speech Detected':
                st.error('Hate Speech')
            elif prediction == 'Offensive Language Detected':
                st.warning('Offensive Speech')
            else:
                st.success('Non-Hate Speech')
        else:
            st.warning('Please enter some text.')

if __name__ == '__main__':
    main()

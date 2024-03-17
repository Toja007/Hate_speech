#                                                                       Hate Speech Detection Model

![download](https://github.com/Toja007/Hate_speech/assets/131866743/a573b5ba-1e2c-4076-9a3e-b4a1f5bb5300)

# Overview
This repository contains a hate speech detection model developed using natural language processing (NLP) techniques. The model is designed to identify and classify hate speech in text data, providing a valuable tool for content moderation and social media analysis.

# Dataset
The hate speech detection model was trained and tested using a labeled dataset consisting of text samples categorized into hate speech, offensive language, and non-hate speech categories. The dataset was preprocessed to handle text normalization, tokenization, and feature extraction before training the classification model.

# Model Architecture
The hate speech detection model is based on a classification model, such as a decision tree, logistic regression, random forest, or deep learning classifier. The model utilizes techniques like word embeddings, feature engineering, and hyperparameter tuning to achieve accurate classification of hate speech, offensive language, and non-hate speech instances.

# Performance Evaluation
The model's performance is evaluated using classification accuracy. These metrics provide insights into the model's ability to correctly classify hate speech, offensive language, and non-hate speech instances.

# Usage
To use the hate speech detection model:

Clone this repository to your local machine.
Install the necessary dependencies, such as Python, scikit-learn, TensorFlow, or PyTorch.
Download or prepare your own text data for hate speech detection.
Preprocess the text data, including normalization, tokenization, and feature extraction.
Load the trained classification model or train a new model using the provided code or scripts.
Use the model to predict hate speech, offensive language, or non-hate speech labels for new text samples.

# Streamlit Application
A Streamlit application named "hate_app.py" is included in this repository. You can run the Streamlit app locally to interactively test the hate speech detection model and visualize its predictions.


<img width="917" alt="Capture" src="https://github.com/Toja007/Hate_speech/assets/131866743/003b4cf8-6dfa-4fc3-bc98-e716cabc434e">


To run the Streamlit app:

Install Streamlit by running pip install streamlit.
Navigate to the directory containing "hate_app.py" in your terminal.
Run the command streamlit run hate_app.py.
Access the Streamlit app in your web browser to use the hate speech detection model.

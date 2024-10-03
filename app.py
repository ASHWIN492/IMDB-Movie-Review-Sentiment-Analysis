import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
from io import BytesIO

# Load the word index and model
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
model = load_model('model.h5')

# Get the vocabulary size from the model
vocab_size = model.get_layer(index=0).input_dim - 1  # Subtract 1 to account for the 0 index

# Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Preprocess text function
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2) + 3, vocab_size) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    pre = preprocess_text(review)
    prediction = model.predict(pre)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, float(prediction[0][0])  # Convert to Python float

# Function to process reviews
def process_reviews(reviews):
    results = []
    for review in reviews:
        sentiment, confidence = predict_sentiment(review)
        results.append({
            'Review': review,
            'Sentiment': sentiment,
            'Confidence': confidence
        })
    return pd.DataFrame(results)

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')

# Text area for multiple reviews
user_reviews_input = st.text_area("Enter movie reviews (one per line):", height=300)

# Button to trigger sentiment analysis
if st.button('Analyze Reviews'):
    if user_reviews_input:
        reviews = user_reviews_input.splitlines()  # Split the input by lines
        results_df = process_reviews(reviews)

        # Display results
        st.write("Results Preview:")
        st.dataframe(results_df)

        # Prepare Excel file for download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Sentiment Analysis')
        excel_data = output.getvalue()

        st.download_button(
            label="Download Results",
            data=excel_data,
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Display overall statistics
        st.write("Overall Statistics:")
        total_reviews = len(results_df)
        positive_reviews = len(results_df[results_df['Sentiment'] == 'Positive'])
        negative_reviews = total_reviews - positive_reviews

        st.write(f"Total Reviews: {total_reviews}")
        st.write(f"Positive Reviews: {positive_reviews} ({positive_reviews/total_reviews:.2%})")
        st.write(f"Negative Reviews: {negative_reviews} ({negative_reviews/total_reviews:.2%})")
    else:
        st.warning("Please enter at least one movie review to analyze.")

# Additional information
st.sidebar.header("About")
st.sidebar.info("This app uses a deep learning model trained on IMDB movie reviews to predict the sentiment of user-input reviews. The model classifies reviews as either positive or negative.")

st.sidebar.header("How to use")
st.sidebar.info(""" 
1. Enter your movie reviews in the text area, one review per line.
2. Click the 'Analyze Reviews' button.
3. The app will display the predicted sentiments and confidence levels.
4. You can download the results as an Excel file.
""")

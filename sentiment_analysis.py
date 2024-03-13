# Importing required packages
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd

# Load spaCy model and add TextBlob to spaCy pipeline for sentiment analysis
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Function to clean text data
def clean_text(text):
    """Clean the text data by removing stop words and punctuation"""
    doc = nlp(text)
    clean_tokens = [token.text.lower().strip() for token in doc if not token.is_stop and token.text.isalpha()]
    return " ".join(clean_tokens)

# Function for sentiment analysis
def analyze_sentiment(review_text):
    """Analyze the sentiment of a given text and return the sentiment score"""
    cleaned_text = clean_text(review_text)
    doc = nlp(cleaned_text)
    polarity = doc._.blob.polarity

    # Return the label and polarity score for the sentiment analysis
    if polarity > 0:
        return 'Positive', polarity
    elif polarity < 0:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

# Load the dataset
df = pd.read_csv('amazon_product_reviews.csv')

# Clean the dataset by removing missing values in 'reviews.text'
clean_data = df.dropna(subset=['reviews.text'])

# Analyze sentiment of the first few reviews
print("Sentiment Analysis of the First Few Reviews:\n")
for index, row in clean_data.head().iterrows():
    sentiment, polarity = analyze_sentiment(row['reviews.text'])
    print(f"Review: {row['reviews.text'][:100]}... | Sentiment: {sentiment} | Polarity: {polarity:.2f}")

# Compare similarity between two reviews
print("\nSimilarity Comparison Between the First Two Reviews:\n")
doc1 = nlp(clean_text(clean_data['reviews.text'].iloc[0]))
doc2 = nlp(clean_text(clean_data['reviews.text'].iloc[1]))
similarity = doc1.similarity(doc2)
print(f"Similarity: {similarity:.2f}")

# Function to test the sentiment analysis model on sample reviews
def test_sample_reviews(sample_reviews):
    """Test the sentiment analysis model on sample reviews and print the results"""
    print("\nSentiment Analysis of Sample Reviews:\n")
    for review in sample_reviews:
        sentiment, polarity = analyze_sentiment(review)
        print(f"Review: {review[:100]}... | Sentiment: {sentiment} | Polarity: {polarity:.2f}")

# Define sample reviews for testing
sample_reviews = [
    "This product was amazing, it worked beyond my expectations.",
    "I'm really disappointed with the purchase. The quality is poor and it broke after a week.",
    "It's okay, not great but not terrible either. Does the job I guess."
]

# Test the model on the sample reviews
test_sample_reviews(sample_reviews)
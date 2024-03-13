# finalCapstone
This code performs sentiment analysis and similarity comparison on a dataset of Amazon
product reviews.
1. Dataset: The dataset, 'amazon_product_reviews.csv', contains information about various
products sold on Amazon.com, US, along with the reviews written by customers.
2. Preprocessing: The code imports the necessary libraries, spacy, spacytextblob, and
pandas, for performing the analysis. The spacy model is loaded and TextBlob is added to the
spaCy pipeline for sentiment analysis. The code has two functions - one to clean the text by
removing stop words and punctuations, and another to perform sentiment analysis and
return the polarity score based on the TextBlob package. Sentiment analysis is performed by
calculating the polarity of each review in the dataset and classifying the output as positive,
negative, or neutral.
3. Results: The sentiment analysis and similarity comparison are carried out on the dataset.
The sentiment analysis is printed for the first few reviews in the dataset, and similarity
comparison is conducted between the first two reviews. Lastly, the model is tested on
sample reviews.
4. Strengths and Limitations: The model performs well in accurately identifying the polarity of
the reviews in the dataset. However, it has limitations in capturing subtler emotions and
tones conveyed in the text. The model can be improved by using more advanced techniques
such as fine-grained sentiment analysis. Additionally, the model can benefit from
incorporating other features such as context, sarcasm, and irony in the analysis

# Start with loading all necessary libraries
import nltk
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pymongo
import pprint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")

print(client.list_database_names())

db = client["Preprocesing"]
print(db.list_collection_names())

collection = db['localTweets']
print(collection.find_one().keys())

cantidad_tweets = collection.count_documents({})
cantidad_tweets_vacios = collection.count_documents({"user.location": ""})
cantidad_ecuador = collection.count_documents(({"user.location": "Ecuador"}))

tweet_texts = set()
no_rt_tweets = []
tweets = collection.find()

for tweet in tweets:
    tweet_text = tweet['full_text']
    if tweet_text not in tweet_texts:
        tweet_texts.add(tweet_text)
        no_rt_tweets.append(tweet)

print(f"-----------------Estadísiticas------------\n"
      f"Cantidad de tweets total: {cantidad_tweets}\n"
      f"User Location Vacía: {cantidad_tweets_vacios}\n"
      f"User Location Ecuador: {cantidad_ecuador}\n"
      f"Cantidad de tweets originales: {len(no_rt_tweets)}")

# Retrieve data from MongoDB
data = list(collection.find())

# Convert data to DataFrame
tweetDF = pd.DataFrame(data)

# Print the DataFrame
print(tweetDF.head())

def preprocess(text):
    text = text.lower()
    text = re.sub('@[A-Za-z0-9_]+', '', text) #remove users
    text = re.sub('[^a-zA-ZáéíóúÁÉÍÓÚ. \s]', '', text) #remove special characters
    text = re.sub('htpps://\S+', '', text) #remove url
    text = re.sub('[^\w\s]', '', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text


def remove_stopwords(text):
    stop_words = stopwords.words('spanish')
    stop_words.extend(['rt'])

    # Split the text into individual words
    words = text.split(' ')

    # Remove stop words from the list of words
    filtered_words = [word for word in words if word not in stop_words]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
    return filtered_text


tweetDF['full_text'] = tweetDF['full_text'].map(lambda x: preprocess(x))
tweetDF['full_text'] = tweetDF['full_text'].map(lambda x: remove_stopwords(x))

print(tweetDF.loc[:, ['id', 'full_text']])


row = tweetDF.loc[tweetDF['id'] == '1652100377121005569']
print(row)
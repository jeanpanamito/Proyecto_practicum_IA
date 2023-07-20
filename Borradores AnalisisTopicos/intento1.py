import gensim
import nltk
import pymongo
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Uri de conexión con Credenciales
uri = "mongodb+srv://mate01:mcxDZa9yU8aUaK2O@cluster0tweet-gp.hkqaqos.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Base de Datos y Colección
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]

# Obtener los documentos de la colección
datos = list(db[mongo_collection].find().limit(5))
tweets = [d['full_text'] for d in datos]

stop_words = stopwords.words('spanish')
stop_words.extend(['rt'])
stop_words.extend(['q'])

def preprocess(text):
    text = text.lower()
    text = re.sub('@[A-Za-z0-9_]+', '', text)  # remove users
    text = re.sub('[^a-zA-ZáéíóúÁÉÍÓÚñ. \s]', '', text)  # remove special characters
    text = re.sub('https?://\S+', '', text)  # remove url
    text = re.sub('[^\w\s]', '', text)  # Remove punctuation
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text

def remove_stopwords(text):
    words = text.split(' ')
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

data_words = list(sent_to_words(tweets))

# Construir los modelos bigrama y trigramas
bigram = Phrases(data_words, min_count=5, threshold=100) # umbral más alto menos frases.
trigram = Phrases(bigram[data_words], threshold=100)

# Forma más rápida de convertir una oración en un trigrama/bigrama
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

# Función para aplicar el preprocesamiento, eliminación de stopwords y lematización
def preprocess_text(text):
    text = preprocess(remove_stopwords(text))
    return lemmatize_text(text)

# Aplicar el preprocesamiento a los tweets
tweetDF = pd.DataFrame(datos)
tweetDF['full_text'] = tweetDF['full_text'].map(preprocess_text)

print(tweetDF[:1]['full_text'])

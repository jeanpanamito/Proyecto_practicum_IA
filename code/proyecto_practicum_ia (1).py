# -*- coding: utf-8 -*-
"""Proyecto_practicum_IA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm

###Instalar Modulos
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install tweepy pymongo
!pip install transformers
!pip install pyspellchecker

"""### Importar librerias"""

import tweepy
import nltk
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
from nltk.corpus import stopwords
from spellchecker import SpellChecker

#descargar una vez
nltk.download('punkt')
nltk.download('stopwords')


#Aquí hay una breve explicación de cada importación:

#1. `tweepy`: Una biblioteca de Python para acceder a la API de Twitter.
#2. `nltk`: La biblioteca Natural Language Toolkit para el procesamiento del lenguaje natural.
#3. `pymongo.mongo_client`: Un módulo para conectarse a una base de datos de MongoDB.
#4. `pymongo.server_api`: Un módulo para especificar la versión de la API del servidor de MongoDB.
#5. `nltk.sentiment.vader`: Un módulo para realizar análisis de sentimientos utilizando el modelo VADER.
#6. `re`: El módulo de expresiones regulares para realizar operaciones de búsqueda y manipulación de cadenas de texto.
#7. `nltk.tokenize.word_tokenize`: Un módulo para dividir un texto en palabras o tokens individuales.
#8. `transformers`: Una biblioteca de Python para usar modelos de aprendizaje automático preentrenados en el procesamiento del lenguaje natural.
#9. `numpy`: Una biblioteca de Python para realizar operaciones numéricas y matemáticas eficientes.
#10. `scipy.special.softmax`: Una función de la biblioteca SciPy para calcular la función softmax en un arreglo.
#11. `csv`: Un módulo para trabajar con archivos CSV (valores separados por comas).
#12. `urllib.request`: Un módulo para abrir y leer URL.
#13. `nltk.corpus.stopwords`: Un módulo que contiene una lista de palabras comunes que se pueden filtrar en el procesamiento del lenguaje natural.
#14. `spellchecker`: Una biblioteca de Python para corregir ortografía en textos.

"""### Conectar la base de datos MongoDB"""

# Uri de conexión con Credenciales
# uri en mongo compass: mongodb+srv://mate01:mcxDZa9yU8aUaK2O@cluster0tweet-gp.hkqaqos.mongodb.net/
uri = "mongodb+srv://mate01:mcxDZa9yU8aUaK2O@cluster0tweet-gp.hkqaqos.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")  #
except Exception as e:
    print(e)

"""### Especificar Coleccion de la base de datos"""

# Base de Datos y Colección
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]

# Consulta de los datos
datos = db.tweets.find()

"""### Filtrado por Ubicación"""

# Diccionario de provincias del Ecuador
ubi_dict = {
    'Ecuador': 'EC',
    'Azuay': 'AZ',
    'Bolivar': 'BO',
    'Cañar': 'CA',
    'Carchi': 'CR',
    'Chimborazo': 'CH',
    'Cotopaxi': 'CO',
    'El Oro': 'EO',
    'Esmeraldas': 'ES',
    'Galapagos': 'GA',
    'Guayas': 'GU',
    'Imbabura': 'IM',
    'Loja': 'LO',
    'Los Rios': 'LR',
    'Manabi': 'MA',
    'Morona Santiago': 'MS',
    'Napo': 'NA',
    'Orellana': 'OR',
    'Pastaza': 'PA',
    'Pichincha': 'PI',
    'Santa Elena': 'SE',
    'Santo Domingo de los Tsachilas': 'SD',
    'Sucumbios': 'SU',
    'Tungurahua': 'TU',
    'Zamora Chinchipe': 'ZC'
}

# Crear una expresión regular para buscar coincidencias de provincias
regex_pattern = re.compile(r'\b(?:' + '|'.join(ubi_dict.keys()) + r')\b', re.IGNORECASE)


# Obtencion de tweets solo de Ecuador sin retweets
ecuador_tweets = []
tweet_texts = set()

for dato in datos:
    tweet_text = dato['full_text']

    if tweet_text not in tweet_texts:
        tweet_texts.add(tweet_text)
        if re.search(regex_pattern, dato['full_text']) \
                or re.search(regex_pattern, dato['user']['location']) \
                or re.search(regex_pattern, ' '.join(dato['hashtags'])):
            ecuador_tweets.append(dato)

"""### Pre-Procesamiento de Datos"""

# Funciones
# Texto a minúsculas
def text_lowercase(text):
    return text.lower()

#Eliminar RT
def remove_rt(text):
    return text.replace("rt", "")

#Eliminar números
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

#Eliminar URL
def remove_url(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)

#Eliminar puntuaciones
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

#Eliminar caracteres especiales
def remove_special_character(text):
    return re.sub(r'[^\w\s]', '', text)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

"""### Preprocesamiento-Tokenizado"""

def tokenize(text):
    # Tokenizar el texto en palabras individuales
    tokens = word_tokenize(text, language='spanish')

    # Eliminar palabras vacías (stop words)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]

    # Unir las palabras nuevamente en un solo texto
    # preprocessed_text = ' '.join(tokens)

    # return preprocessed_text
    return tokens

"""###Analisis de Sentimiento

"""

import nltk

# Crear una instancia del analizador de sentimientos VADER
sia = SentimentIntensityAnalyzer()

datos = db[mongo_collection].find()

for tweet in datos:
    tweet_text = tweet['full_text']

    # Convertir a minúsculas, quitar números, "rt" y puntuación
    clean_text = remove_punctuation(remove_rt(remove_numbers(text_lowercase(tweet_text))))

    # Tokenización del texto utilizando una expresión regular
    tokens = re.findall(r'\w+', clean_text)

    # Etiquetar el sentimiento de cada token utilizando VADER
    sentiment_scores = [sia.polarity_scores(token)['compound'] for token in tokens]

    # Asignar el sentimiento promedio del texto al campo 'sentiment'
    tweet['sentiment'] = sum(sentiment_scores) / len(sentiment_scores)

    # Presentar los datos tokenizados y etiquetados
    print("Texto original:", tweet_text)
    print("Clean Text:", clean_text)
    print("Tokens:", tokens)
    print("Sentimientos:", sentiment_scores)
    print("Sentimiento promedio:", tweet['sentiment'])
    print("------------------------")

"""###API Twitter-roBERTa-base for Sentiment Analysis"""

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night 😊"

text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

"""### Presentación de Datos"""

spell = SpellChecker(language='es')
# Preprocesado de los tweets de Ecuador
for ecutweet in ecuador_tweets:
    full_text = ecutweet['full_text']
    id = ecutweet['id']
    print(f'------------Full Text - {id} ------------\n {full_text}')

    # Convertir a minusculas, numeros, rt y puntuacion
    n1 = str(preprocess
             (remove_punctuation
               (remove_url
                (remove_numbers
                 (remove_rt
                  (text_lowercase(full_text)))))))

    #Tokenizar
    words = tokenize(n1)

    #PyspellChecker 'Prueba001'
    for word in words:
        # Check si la palabra está mal escrita
        if not spell.correction(word) == word:
            # Obtener la correción más probable
            correction = spell.correction(word)
            #
            print(f"The word '{word}' is misspelled. Did you mean '{correction}'?")

    print(f'---------------Clean Text----------------\n {words}')
    #break
print(len(ecuador_tweets))
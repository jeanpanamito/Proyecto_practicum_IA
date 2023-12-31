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
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

"""### Especificar Coleccion de la base de datos"""

# Base de Datos y Colección
db = client["Preprocessing"]
collection = db['tweets']

# Comprobación de Campos
collection.find_one().keys()

"""### Estadísticas básicas"""

tweets = collection.find()

cantidad_tweets = collection.count_documents({})
cantidad_tweets_vacios = collection.count_documents({"user.location": ""})
cantidad_ecuador = collection.count_documents(({"user.location": "Ecuador"}))

#Proceso para sacar tweets en que su texto es el mismo (retweets)
tweet_texts = set()
no_rt_tweets = []
for tweet in tweets:
    tweet_text = tweet['full_text']
    if tweet_text not in tweet_texts:
        tweet_texts.add(tweet_text)
        no_rt_tweets.append(tweet)

print(f"---------------Estadísiticas Básicas------------\n"
      f"Cantidad de tweets total: {cantidad_tweets}\n"
      f"User Location Vacía: {cantidad_tweets_vacios}\n"
      f"User Location Ecuador: {cantidad_ecuador}\n"
      f"Cantidad de tweets originales: {len(no_rt_tweets)}")

"""### Dataframe para manejo de los datos"""

import pandas as pd

# Retrieve data from MongoDB
data = list(collection.find())
tweetDF = pd.DataFrame(data)

import pandas as pd
# Obtener data de MongoDB
pruebasDF = pd.DataFrame(data)

# Agrupar usuarios por ubicación
grouped_data = pruebasDF.groupby(pruebasDF['user'].apply(lambda x: x['location']))

# Imprimir columnas grupo, id
for group_name, group_data in grouped_data:
    print("Location:", group_name)
    print(group_data[['id']])
    break

# Top 10 países con mayor frecuencia

# Crear diccionario vacío
location_freq = {}

# Recorrer los grupos y obtener la frecuencia
for group_name, group_data in grouped_data:
    location = group_name
    count = len(group_data)
    location_freq[location] = count

sorted_locations = sorted(location_freq.items(), key=lambda x: x[1], reverse=True)

# Obtener las 10 locaciones con mayor frencuencia
top_10_locations = sorted_locations[:10]

# Presentar
for location, count in top_10_locations:
    print("Location:", location)
    print("Count:", count)

# Cantidad de locaciones
len(location_freq.keys())

# Locaciones que coiciden con la palabra Ecuador
keyword = "Ecuador"

# Crea una lista para almacenar las claves que coinciden con la palabra clave
matching_keys = []

# Recorre las claves del diccionario y verifica si contienen la palabra clave
for location in location_freq.keys():
    if keyword.lower() in location.lower():
        matching_keys.append(location)

# Imprime la lista de claves que coinciden
print(matching_keys)
print(len(matching_keys))

import pandas as pd

# Filtrar tweets con las locaciones obtenidas en base a la palabra Ecuador

# Crear un DataFrame vacío para almacenar los tweets filtrados
tweets_filtrados = pd.DataFrame()

# Recorrer las filas del DataFrame original y filtrar por ubicación
for index, row in pruebasDF.iterrows():
    ubicacion = row['user']['location']
    if any(keyword.lower() in ubicacion.lower() for keyword in matching_keys):
        tweets_filtrados = pd.concat([tweets_filtrados, pd.DataFrame(row).T])

# Restablecer los índices del DataFrame filtrado
tweets_filtrados.reset_index(drop=True, inplace=True)

# Imprimir el DataFrame filtrado
print(len(tweets_filtrados))

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

long_string = ' '.join(list(tweetDF['clean_full_text'].values))

# Obtener las stopwords en español de NLTK
stopwords_sp = set(stopwords.words('spanish'))

# Agregar "rt" como una stop word adicional
stopwords_sp.add('rt')

wordcloud = WordCloud(
    background_color="white", max_words=5000, contour_color='steelblue',
    contour_width=3, stopwords=stopwords_sp)
wordcloud.generate(long_string)
wordcloud.to_image()

#La columna clean full text eliminó letras con tildes

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

"""### Preprocesamiento de Datos"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stop_words = stopwords.words('spanish') #Idioma para Stopwords
stop_words.extend(['rt']) #Añadir RT como stopword

def preprocess(text):
    text = text.lower()
    text = re.sub('@[A-Za-z0-9_]+', '', text) #remover usuarios
    text = re.sub('[^a-zA-ZáéíóúÁÉÍÓÚñ. \s]', '', text) #remover caracteres especiales manteniendo acentos
    text = re.sub('htpps://\S+', '', text) #remover url
    text = re.sub('[^\w\s]', '', text)  # remover puntuaciones
    text = re.sub('\s+', ' ', text)  # remover extra espacios
    text = text.strip()  # Remover leading/trailing spaces
    return text

# Opcion de devolverlo como un solo String
def remove_stopwords(text):
    # Split the text into individual words
    words = text.split(' ')
    # Remove stop words from the list of words
    filtered_words = [word for word in words if word not in stop_words]
    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

"""### Preprocesamiento -Tokenizado"""

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

"""### Aplicación de Funciones"""

tweetDF['full_text'] = tweetDF['full_text'].map(lambda x: preprocess(x))
tweetDF['full_text'] = tweetDF['full_text'].map(lambda x: remove_stopwords(x))
tweetDF['full_text'] = tweetDF['full_text'].map(lambda x: tokenize(x))
#Imprimir solo columnas que nos interesan
print(tweetDF.loc[:, ['id', 'full_text']])

tweetDF.loc[1:, ['id', 'full_text']]

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Convertir cada lista de palabras en una cadena de texto
text_list = [' '.join(word_list) for word_list in tweetDF['full_text'].values]

# Unir todas las cadenas en una sola cadena
long_string = ' '.join(text_list)

wordcloud = WordCloud(background_color="white", max_words=7000, contour_color='steelblue', contour_width=3)
wordcloud.generate(long_string)
wordcloud.to_image()

#Obtención de una mejor imagen

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

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

for tweet in datos:
    tweet_text = tweet['full_text']

    text = preprocess(tweet_text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print("Texto original:", tweet_text + f"\n{i+1}) {l} {np.round(float(s), 4)}")

    print("------------------------")

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
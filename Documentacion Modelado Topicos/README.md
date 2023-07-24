# MODELADO DE TOPICOS
## Instalación de módulos
En este proyecto, se han importado diversas librerías y módulos especializados para potenciar el procesamiento del lenguaje natural y el modelado de tópicos. Lo que nos permite tener un análisis exhaustivo de los datos textuales y la extracción de temas relevantes de manera eficiente.

```python
!pip install tweepy pymongo
!pip install transformers
```

## Importacion de librerias

Las librerias más importantes y necesarias para el proyecto fueron: 

- NLTK: proporciona una amplia gama de herramientas para el procesamiento del lenguaje natural.
- Gensim: utilizada para modelado de tópicos y procesamiento de texto.
- NumPy y pandas: son esenciales para el procesamiento eficiente de datos numéricos y estructurados. 
- Matplotlib: se utiliza para visualizar los resultados del modelado de temas, lo que facilita la interpretación de los resultados obtenidos.
  
```pyhton
import numpy as np
import gensim
import nltk
import pymongo
from gensim import corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import re
from pprint import pprint
from gensim import corpora,models
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
## Conexión a la Base de Datos
```python
uri = "mongodb+srv://mate01:mcxDZa9yU8aUaK2O@cluster0tweet-gp.hkqaqos.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```

## Preprocesamiento
Durante esta fase, se aplico una **limpieza** a los datos, la cual consiste en remover caracteres especiales, signos de puntuación, espacios y todo lo que era irrelevante para nuestro ánalisis, luego realizamos la **tokenización** para convertir el texto en una secuencia de elementos discretos que pueden ser analizados y procesados por algoritmos de PLN o modelos de aprendizaje automático, para posteriormente usar el **lematizado** con el intento de reducir cada palabra a su forma base o "lemma".
Dentro de una nueva colección llamada ecuadorTweets, se cargaron los datos ya preprocesados para acceder a ellos con una mayor eficiencia, dentro de la colección trabajaremos con el atributo clean_text el cual tendra la información de cada tweet.

```python
mongo_db = 'Preprocessing'
mongo_collection = 'ecuadorTweets'
db = client[mongo_db]

datos = list(db[mongo_collection].find().limit(100))
tweets = [d['clean_text'] for d in datos]

stop_words = stopwords.words('spanish')
stop_words.extend(['rt'])
stop_words.extend(['q'])
```



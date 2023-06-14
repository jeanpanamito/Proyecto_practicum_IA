# Proyecto_practicum_IA
Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional
# Documentación del proyecto "Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional"

El proyecto "Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional" utiliza varias técnicas de procesamiento del lenguaje natural y análisis de sentimientos para procesar y analizar datos relacionados con la seguridad nacional.

## Descripción de Metadatos
Se utilizó la API de Twitter para guardar en una colección de MongoDB documentos que contienen metadatos de tweets sobre Seguridad en Ecuador
Metadato | Descripción |
---------| ----------- |
id | Codigo del tweet
time | Fecha y hora de extracción del tweet 
created_at | Fecha y hora de creación del tweet
full_text | Texto completo del tweet
clean_full_text | Texto procesado del tweet
user | Es un objeto del usuario que contiene su nombre, ubicación, numero de seguidores
url_tweet | Url construida de acceso al tweet
place | Lugar de creación del tweet
retweet_count | Conteo de retweets
hastags | Hastags extraidos del tweet
urls | Urls extraidas del tweet
photos | Urls de imágenes que contiene el tweet
videos | Urls de videos que contenga el tweet

## Instalación de módulos
El proyecto requiere la instalación de los siguientes módulos:

```python
!pip install tweepy pymongo
!pip install transformers
!pip install pyspellchecker
```

## Importación de librerías
El proyecto utiliza las siguientes librerías:

```python
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
```

A continuación, se detalla la funcionalidad de cada librería importada:

1. `tweepy`: Proporciona una interfaz para acceder a la API de Twitter.
2. `nltk`: Es la biblioteca Natural Language Toolkit utilizada para el procesamiento del lenguaje natural.
3. `pymongo.mongo_client`: Se utiliza para conectarse a una base de datos de MongoDB.
4. `pymongo.server_api`: Se utiliza para especificar la versión de la API del servidor de MongoDB.
5. `nltk.sentiment.vader`: Proporciona una herramienta para realizar análisis de sentimientos utilizando el modelo VADER.
6. `re`: El módulo de expresiones regulares se utiliza para realizar operaciones de búsqueda y manipulación de cadenas de texto.
7. `nltk.tokenize.word_tokenize`: Se utiliza para dividir un texto en palabras o tokens individuales.
8. `transformers`: Es una biblioteca para usar modelos de aprendizaje automático preentrenados en el procesamiento del lenguaje natural.
9. `numpy`: Se utiliza para realizar operaciones numéricas y matemáticas eficientes.
10. `scipy.special.softmax`: Proporciona una función para calcular la función softmax en un arreglo.
11. `csv`: Se utiliza para trabajar con archivos CSV (valores separados por comas).
12. `urllib.request`: Se utiliza para abrir y leer URL.
13. `nltk.corpus.stopwords`: Proporciona una lista de palabras comunes que se pueden filtrar en el procesamiento del lenguaje natural.
14. `spellchecker`: Es una biblioteca para corregir la ortografía en textos.

## Conexión a la base de datos MongoDB
El proyecto se conecta a una base de datos MongoDB utilizando la siguiente configuración:

```python
uri = "mongodb+srv://<username>:<password>@<cluster_url>/"
client = MongoClient(uri, server_api=ServerApi('1'))
```

El URI de conexión debe reemplazarse con las credenciales y la URL del clúster MongoDB.

## Especificación de la colección de la base de datos
El proyecto especifica la base de datos y la colección de MongoDB con la siguiente configuración:

```python
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]
datos = db[mongo_collection].find
```


## Preprocesamiento de datos

El proyecto realiza varias etapas de preprocesamiento de datos para preparar los textos de los tweets antes de realizar el análisis de sentimientos y la clasificación.

### 1. Limpieza de texto

Se realiza una serie de pasos para limpiar el texto de los tweets:

- Se eliminan los enlaces URL utilizando expresiones regulares.
- Se eliminan las menciones a otros usuarios de Twitter.
- Se eliminan los caracteres especiales y los números.
- Se convierten todos los caracteres a minúsculas.
- Se eliminan los signos de puntuación.
- Se eliminan los espacios en blanco adicionales.

### 2. Tokenización

Los tweets se dividen en palabras o tokens individuales utilizando el tokenizador `word_tokenize` de NLTK. Esto nos permite trabajar con cada palabra por separado en etapas posteriores.

### 3. Eliminación de palabras irrelevantes

Se eliminan las palabras irrelevantes, como los artículos, pronombres y preposiciones, utilizando la lista de palabras vacías (stop words) proporcionada por NLTK.

### 4. Corrección ortográfica

Se realiza una corrección ortográfica en los tweets utilizando la biblioteca `SpellChecker`. Esto ayuda a corregir posibles errores de escritura y mejorar la precisión del análisis de sentimientos.

### 5. Lemmatización

Se realiza la lematización de las palabras para reducir las palabras a su forma base o lema. Esto ayuda a reducir la variabilidad y mejorar la precisión del análisis de sentimientos.

## Modelado de tópicos

Durante esta etapa, se pretende recuperar toda la información obtenida desde los tweets para encontrar los temas generales y temáticas implicitas que abarquen todo el contenido del texto, para de esa forma ordenar, resumir y tener mejor comprensión sobre el mismo.
Para este proyecto, hemos decidido trabajar bajo el algoritmo de LDA(Latent Dirichlet Allocation).

## Análisis de sentimientos

El proyecto utiliza el modelo VADER (Valence Aware Dictionary and sEntiment Reasoner) para realizar el análisis de sentimientos de los tweets. VADER es un modelo de análisis de sentimientos específicamente diseñado para el análisis de texto social, como los tweets. Proporciona una puntuación de sentimiento compuesta que indica la polaridad del texto (positivo, negativo o neutro).

## Clasificación de temas

El proyecto utiliza un modelo de clasificación de temas preentrenado para clasificar los tweets en categorías temáticas. El modelo utiliza la arquitectura BERT (Bidirectional Encoder Representations from Transformers) y ha sido entrenado en un conjunto de datos etiquetados. El modelo toma el texto del tweet y lo clasifica en una de las categorías temáticas predefinidas.

## Almacenamiento de resultados

Los resultados del análisis de sentimientos y clasificación de temas se almacenan en la base de datos MongoDB. Cada documento en la colección "tweets" contiene la información original del tweet, el sentimiento calculado y la categoría temática asignada.

## Uso del proyecto

El proyecto se puede utilizar para procesar y analizar datos relacionados con la seguridad nacional en Twitter. Para utilizarlo, se deben seguir los siguientes pasos:

1. Configurar la conexión a la base de datos MongoDB, especificando las credenciales y la URL del clúster.
2. Ejecutar el preprocesamiento de datos para limpiar, tokenizar y procesar los textos de los tweets.
3. Realizar el análisis de sentimientos utilizando el modelo VADER para obtener la puntuación de sentimiento de cada tweet.
4. Realizar la clasificación de temas utilizando el modelo preentrenado para asignar categorías temáticas a los tweets.
5. Almacenar los resultados en la base de datos MongoDB para su posterior análisis.

## Conclusiones

El proyecto proporciona una manera eficiente de procesar y analizar grandes volúmenes de datos de Twitter relacionados con la seguridad nacional. El preprocesamiento de datos y el análisis de sentimientos permiten obtener información valiosa sobre la opinión pública y los temas relevantes en la plataforma. La clasificación de temas ayuda a organizar y categorizar los tweets, facilitando el análisis y la identificación de tendencias.

El proyecto se puede adaptar y personalizar según las necesidades específicas del usuario, como agregar nuevas categorías temáticas o entrenar modelos personalizados. También se pueden aplicar técnicas adicionales, como la detección de entidades y la extracción de características, para obtener información más detallada de los tweets.

En resumen, el proyecto ofrece una solución robusta y escalable para el análisis de datos de Twitter relacionados con la seguridad nacional, brindando una comprensión más profunda de las opiniones y temas relevantes en la plataforma.

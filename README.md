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
El proyecto requiere la instalación de los siguientes módulos. El signo de exclamación es en caso de realizarse en un notebook. La línea de código: 
`-m spacy download es_core_news_sm` descarga un modelo específico para el procesamiento del lenguaje español (es) proporcionado por Spacy. 

```shell
!pip install pandas 
!pip install tweepy pymongo 
!pip install spacy 
!python -m spacy download es_core_news_sm #Setting adicional para Spacy
!pip install transformers
!pip install pyspellchecker
!pip install openai
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
import spacy
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
15. `spacy`: Biblioteca de procesamiento del lenguaje natural (PLN). Se la utilizará para 

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


## [Preprocesamiento](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=omi0VcgIhMpP) {:target="_blank"}

El proyecto realiza varias etapas de preprocesamiento de datos para preparar los textos de los tweets antes de realizar el análisis de sentimientos y la clasificación.


### 1. Limpieza de texto

Se realiza una serie de pasos para limpiar el texto de los tweets:

- Remover usuarios
- Remover caractéres especiales manteniendo acentos.
- Remover URL.
- Remover puntuaciones.
- Remover espacios extra.
- Remover leading/trailing spaces.

### 2. StopWords
Se usa la librería nltk para remover stopwords <br>
Se descargan los recursos para el tokenizador y las palabras vacías (stopwords) en español utilizando `nltk.download('stopwords')` y `nltk.download('punkt')` <br>
Se carga la lista de palabras vacías en español utilizando stopwords.words('spanish').
Se agrega manualmente la palabra 'rt' a la lista de stopwords utilizando stop_words.extend(['rt']). 'rt' generalmente se refiere a "retweet" y a menudo se elimina en análisis de texto.

### 3. Tokenización

Los tweets se dividen en palabras o tokens individuales utilizando el tokenizador `word_tokenize` de NLTK. Esto nos permite trabajar con cada palabra por separado en etapas posteriores.

### 4. Eliminación de palabras irrelevantes

Se eliminan las palabras irrelevantes, como los artículos, pronombres y preposiciones, utilizando la lista de palabras vacías (stop words) proporcionada por NLTK.

### 5. Corrección ortográfica

Se realiza una corrección ortográfica en los tweets utilizando la biblioteca `SpellChecker`. Esto ayuda a corregir posibles errores de escritura y mejorar la precisión del análisis de sentimientos. (No implementada)

### 6. Lemmatización

Se realiza la lematización de las palabras para reducir las palabras a su forma base o lema. Esto ayuda a reducir la variabilidad y mejorar la precisión del análisis de sentimientos.

## Modelado de tópicos

Durante esta etapa, se pretende recuperar toda la información obtenida desde los tweets para encontrar los temas generales y temáticas implicitas que abarquen todo el contenido del texto, para de esa forma ordenar, resumir y tener mejor comprensión sobre el mismo.
Para este proyecto, hemos decidido trabajar bajo el algoritmo de LDA(Latent Dirichlet Allocation).

##  Análisis de Sentimientos 

En la presente documentacion, se describe el proceso de análisis de sentimientos realizado utilizando dos modelos de procesamiento de lenguaje natural. El objetivo fue evaluar diferentes enfoques y determinar cuál de los modelos proporcionaba los mejores resultados en términos de clasificación de sentimientos en textos, específicamente en tweets.

### Modelos Utilizados

Se seleccionaron los siguientes modelos para realizar el análisis de sentimientos:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner):** Se trata de un analizador de sentimientos basado en reglas, incluido en la biblioteca NLTK (Natural Language Toolkit). Este modelo utiliza un conjunto de palabras y reglas predefinidas para asignar un puntaje de sentimiento a cada palabra y generar un puntaje general de sentimiento para un texto.

2. **Twitter-roBERTa-base:** Es un modelo basado en transformer, específicamente diseñado para tareas de análisis de sentimientos en tweets. Se utiliza la biblioteca Transformers para cargar y utilizar este modelo preentrenado.

### Proceso de Análisis de Sentimientos

El proceso de análisis de sentimientos se llevó a cabo de la siguiente manera:

1. **Análisis de Sentimientos con VADER:**

   ```python
   import nltk
   from nltk.sentiment import SentimentIntensityAnalyzer

   # Crear una instancia del analizador de sentimientos VADER
   sia = SentimentIntensityAnalyzer()

   datos = db[mongo_collection].find()

   for tweet in datos:
       tweet_text = tweet['full_text']

       # Realizar preprocesamiento del texto
       clean_text = remove_punctuation(remove_rt(remove_numbers(text_lowercase(tweet_text))))

       # Tokenización del texto utilizando una expresión regular
       tokens = re.findall(r'\w+', clean_text)

       # Etiquetar el sentimiento de cada token utilizando VADER
       sentiment_scores = [sia.polarity_scores(token)['compound'] for token in tokens]

       # Asignar el sentimiento promedio del texto al campo 'sentiment'
       tweet['sentiment'] = sum(sentiment_scores) / len(sentiment_scores)

       # Presentar los resultados
       print("Texto original:", tweet_text)
       print("Clean Text:", clean_text)
       print("Tokens:", tokens)
       print("Sentimientos:", sentiment_scores)
       print("Sentimiento promedio:", tweet['sentiment'])
       print("------------------------")
   ```

   En este proceso, se utilizó el analizador de sentimientos VADER de NLTK. Se realizó un preprocesamiento del texto para eliminar puntuación, números y menciones a retweets. Luego, se tokenizó el texto y se asignó un puntaje de sentimiento a cada token utilizando VADER. El sentimiento promedio se asignó al campo 'sentiment' en cada documento de tweet.

2. **Análisis de Sentimientos con Twitter-roBERTa-base:**

   ```python
   from transformers import AutoModelForSequenceClassification
   from transformers import AutoTokenizer
   import numpy as np
   from scipy.special import softmax

   task = 'sentiment'
   MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

   tokenizer = AutoTokenizer.from_pretrained(MODEL)

   # Cargar el modelo preentrenado
   model = AutoModelForSequenceClassification.from_pretrained(MODEL)
   model.save_pretrained(MODEL)

   text = "Good night 😊"
   text = preprocess(text)
   encoded_input = tokenizer(text, return_tensors='pt')
   output = model(**encoded_input)
   scores = output[0][0].detach().numpy()
   scores = softmax(scores)

   ranking = np.argsort(scores)
   ranking = ranking[::-1]
   for i in range(scores.shape[0]):
       l = labels[ranking[i]]
       s = scores[ranking[i]]
       print(f"{i+1}) {l} {np.round(float(s), 4)}")
   ```

   En este caso, se utilizó el modelo Twitter-roBERTa-base para el análisis de sentimientos en tweets. Se cargó el modelo y se utilizó un tokenizer específico para este modelo. Luego, se realizó el análisis de sentimientos en un texto de ejemplo y se obtuvieron las puntuaciones de sentimiento para cada etiqueta posible.

### Conclusiones

El análisis de sentimientos realizado utilizando los modelos VADER y Twitter-roBERTa-base proporcionó resultados interesantes. El enfoque basado en reglas de VADER fue rápido y fácil de implementar, y mostró una buena capacidad para identificar el sentimiento general en los tweets. Por otro lado, el modelo Twitter-roBERTa-base, basado en transformer, mostró un enfoque más sofisticado y ajustado específicamente para el análisis de sentimientos en tweets.

En general, ambos modelos fueron útiles para el análisis de sentimientos en textos, pero el modelo Twitter-roBERTa-base demostró una mayor precisión y capacidad para capturar matices en los sentimientos expresados en los tweets. Por lo tanto, se recomienda utilizar este modelo para tareas de análisis de sentimientos en tweets en situaciones donde se requiera un mayor nivel de detalle y precisión.

## Clasificación de temas

El proyecto utiliza un modelo de clasificación de temas preentrenado para clasificar los tweets en categorías temáticas. El modelo utiliza la arquitectura BERT (Bidirectional Encoder Representations from Transformers) y ha sido entrenado en un conjunto de datos etiquetados. El modelo toma el texto del tweet y lo clasifica en una de las categorías temáticas predefinidas.

## Almacenamiento de resultados

Los resultados del análisis de sentimientos y clasificación de temas se almacenan en la base de datos MongoDB. Cada documento en la colección "tweets" contiene la información original del tweet, el sentimiento calculado y la categoría temática asignada.

## Conclusiones

El proyecto proporciona una manera eficiente de procesar y analizar grandes volúmenes de datos de Twitter relacionados con la seguridad nacional. El preprocesamiento de datos y el análisis de sentimientos permiten obtener información valiosa sobre la opinión pública y los temas relevantes en la plataforma. La clasificación de temas ayuda a organizar y categorizar los tweets, facilitando el análisis y la identificación de tendencias.

El proyecto se puede adaptar y personalizar según las necesidades específicas del usuario, como agregar nuevas categorías temáticas o entrenar modelos personalizados. También se pueden aplicar técnicas adicionales, como la detección de entidades y la extracción de características, para obtener información más detallada de los tweets.

En resumen, el proyecto ofrece una solución robusta y escalable para el análisis de datos de Twitter relacionados con la seguridad nacional, brindando una comprensión más profunda de las opiniones y temas relevantes en la plataforma.

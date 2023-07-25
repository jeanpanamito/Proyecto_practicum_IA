# Proyecto_practicum_IA
Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional

# Documentación del proyecto "Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional"
El proyecto "Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional" utiliza varias técnicas de procesamiento del lenguaje natural y análisis de sentimientos para procesar y analizar datos relacionados con la seguridad nacional. <br>

Se utilizó el siguiente [notebook](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=nADt1RBGXbDX&uniqifier=1) de Google Colab para el desarrollo del proyecto

## Tabla de Contenidos
  - [Descripción de Metadatos](#descripción-de-metadatos)
  - [Instalación de módulos](#instalación-de-módulos)
  - [Importación de librerías](#importación-de-librerías)
  - [Conexión a la base de datos MongoDB](#conexión-a-la-base-de-datos-mongodb)
  - [Especificación de la colección de la base de datos](#especificación-de-la-colección-de-la-base-de-datos)
  - [Preprocesamiento](#preprocesamiento)
    - [1. Discriminar Retweets](#1-discriminar-retweets)
    - [2. Filtrado por ubicación](#2-filtrado-por-ubicación)
    - [3. Limpieza de texto](#3-limpieza-de-texto)
    - [4. StopWords](#4-stopwords)
    - [5. Tokenización](#5-tokenización)
    - [6. Corrección ortográfica](#6-corrección-ortográfica)
    - [7. Lemmatización](#7-lemmatización)
    - [8. WordCloud](#8-wordcloud)
  - [Modelado de tópicos](#modelado-de-tópicos)
    - 
  - [Análisis de Sentimientos](#análisis-de-sentimientos)
    - [Modelos Utilizados](#modelos-utilizados)
    - [Proceso de Análisis de Sentimientos](#proceso-de-análisis-de-sentimientos)
    - [Conclusiones](#conclusiones)
  - [Clasificación de temas](#clasificación-de-temas)
  - [Almacenamiento de resultados](#almacenamiento-de-resultados)
  - [Conclusiones](#conclusiones-1)



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

## [Instalación de módulos](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=O6QlgIXNOkJO)
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

## [Conexión a la base de datos MongoDB](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=Cqa8eXtVXZQR)
El proyecto se conecta a una base de datos MongoDB utilizando la siguiente configuración:

```python
uri = "mongodb+srv://<username>:<password>@<cluster_url>/"
client = MongoClient(uri, server_api=ServerApi('1'))
```

El URI de conexión debe reemplazarse con las credenciales y la URL de la base deseada, en este caso se usó un cluster de Mongo Atlas.
<br>
Con las siguiente línea de código obtenemos una confirmación en caso de conectarse a una base en la nube
```python
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
```
## Especificación de la colección de la base de datos
El proyecto especifica la base de datos y la colección de MongoDB con la siguiente configuración:

```python
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]
datos = db[mongo_collection].find
```
#### [Colecciones:](https://github.com/jeanpanamito/Proyecto_practicum_IA/tree/main/Archivos)
1. tweets = tweets de muestra Original
2. tweetsOriginals = twets sin rts 
3. tweetsPreprocessed = muestra sin rts con preprocesamiento inicial 
4. tweetsLemmaComparation = tweets sin rts comparacion entre lemmatización nltk y spacy 
5. ecuadorTweets = tweets filtrados y preprocesados
 
## [Preprocesamiento](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=omi0VcgIhMpP)
El proyecto realiza varias etapas de preprocesamiento de datos para preparar los textos de los tweets antes de realizar el análisis de sentimientos y la clasificación.

### [1. Discriminar Retweets](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=akvFu2sjhnCO&line=2&uniqifier=1)
Filtramos y descartamos tweets que son retweets, es decir, aquellos tweets que tienen el mismo texto que otro tweet ya presente en la lista. El objetivo es obtener una lista de tweets que contenga solo los tweets originales sin duplicados. <br>
Así evitamos alteraciones en los resultados por tweets que no aportan texto adicional.

### [2. Filtrado por ubicación](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=mwdTQC2pdlYK&line=1&uniqifier=1)
Se utiliza la expresión `dfProcessed['user'].apply(lambda x: 'ecuador' in x.get('location', '').lower())` para crear una serie booleana que indica si la ubicación de cada usuario en `dfProcessed` contiene la palabra **"ecuador"** en letras minúsculas.
La expresión `x.get('location', '')` se utiliza para obtener el valor de la clave **'location'** del diccionario **x**, que representa cada fila en la columna **'user'** del DataFrame. Si no existe **'location'**, se devuelve una cadena vacía.
El resultado del filtro se almacena en un nuevo DataFrame llamado **filtered_df**.

### [3. Limpieza de texto](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=nWi8u6whhgn9&line=4&uniqifier=1)
Se realiza una serie de pasos para limpiar el texto de los tweets:
- Remover usuarios
- Remover caractéres especiales manteniendo acentos.
- Remover URL.
- Remover puntuaciones.
- Remover espacios extra.
- Remover leading/trailing spaces.

### [4. StopWords](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=FpIHQEKslLEj&line=3&uniqifier=1)
Usamos la librería nltk para remover stopwords <br>
Se descargan los recursos para el tokenizador y las palabras vacías (stopwords) en español utilizando `nltk.download('stopwords')` y `nltk.download('punkt')` <br>
Cargamos la lista de palabras vacías en español utilizando `stopwords.words('spanish')`.
Se agrega manualmente la palabra 'rt' a la lista de stopwords utilizando `stop_words.extend(['rt'])`. 'rt' generalmente se refiere a "retweet" y a menudo se elimina en análisis de texto.

### [5. Tokenización](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=jjCubgkMhjHk&line=4&uniqifier=1)
Los tweets se dividen en palabras o tokens individuales utilizando el tokenizador `word_tokenize` de NLTK. Esto nos permite trabajar con cada palabra por separado en etapas posteriores. Recordar configurar en base al idioma con: `word_tokenize(text, language='spanish')`

### [6. Corrección ortográfica](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=3W22i8mXmz6I&line=7&uniqifier=1)
Se realiza una corrección ortográfica en los tweets utilizando la biblioteca `SpellChecker`. Esto ayuda a corregir posibles errores de escritura y mejorar la precisión del análisis de sentimientos. (No implementada)

### [7. Lemmatización](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=aQyuaRnJbHp4)
Se realiza la lematización de las palabras para reducir las palabras a su forma base o lema. Esto ayuda a reducir la variabilidad y mejorar la precisión del análisis de sentimientos. <br>
Realizamos pruebas con la librería NLTK y Spacy

### [8. WordCloud](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=1tKzw69LB15N&line=9&uniqifier=1)
Utilizamos la biblioteca matplotlib y wordcloud para crear una nube de palabras. <br> 
Se define una lista llamada `text_list`, donde se almacena el contenido limpio de cada tweet como una cadena de texto. 
Se utiliza una comprensión de listas para recorrer la columna `clean_text` de `tweetDF` **(un DataFrame)** y unir las palabras limpias en cada tweet con un espacio para formar una cadena. <br>
Luego, se crea una cadena de texto llamada `long_string`, que une todas las cadenas de texto de `text_list` en una sola cadena. <br>
Se configura el objeto **WordCloud** con algunas opciones, como el color de fondo, el número máximo de palabras a mostrar (max_words), el color y el ancho del contorno.<br>
Se genera la nube de palabras utilizando el método `generate(long_string)` aplicado al objeto **wordcloud**.<br>
Finalmente, la imagen de la nube de palabras se visualiza utilizando el método `to_image()`. 

## [Modelado de tópicos](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=2qnst2d3Zpz7)

Durante esta etapa, se pretende recuperar toda la información obtenida desde los tweets para encontrar los temas generales y temáticas implicitas que abarquen todo el contenido del texto, para de esa forma ordenar, resumir y tener mejor comprensión sobre el mismo.
Para este proyecto, hemos decidido trabajar bajo el algoritmo de LDA(Latent Dirichlet Allocation).

### Exploración de parámetros para el modelo LDA

Se realizo algunos experimentos al momento de definir los valores de Alpha y Beta
- El primero fue usando los valores por defecto, pero la coherencia que obteniamos, era baja.
- El segundo lo realizamos con un programa el cual nos permitia el que el valor de Alpha y Beta se situe en un rango de (0.01, 1, 0.3), dicho resultado lo almacenabamos en un CSV, el cual lo analizamos y escogimos los valores donde cuya coherencia era la más optima, donde el valor que definimos para estos parámetros fue de 0.01.

De la misma manera, luego de definir los valores de Alpha y Beta, necesitabamos encontrar el número de tópicos más favorable para que no exista una duplicación de temas entre ellos, es por eso que generamos una gráfica la cual nos permitia observar el valor de la coherencia junto con el número de tópicos.

- El pico de la gráfica era con 2 tópicos, pero fue descartado ya que era un número muy limitante a relación con una extensa información de los tweets.
- El siguiente más alto, fue 6 tópicos, el cual lo elegimos, debido a que los temas de cada tópico no eran repetitivos y no se intersecaban.

### Selección y experimentación con parámetros óptimos
Al obtener los parámetros óptimos los cuales eran:
- Alpha: 0.01
- Beta: 0.01
- Diccionario: id2word
- Número de tópicos: 6

Se procedio a realizar el modelo final LDA:

```python
num_topics = 6

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.01,
                                           eta=0.01)
```
### Resultado del modelo LDA
Presentamos el resultado del modelo, mediante una gráfica usando la libreria de pyLDAvis.
![Topicos](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/pictures/Topicos.png)

Donde nos permitio observar el contenido de temas que tenía cada tópico y la distancia/diferencia entre tópicos que existe.

##  [Análisis de Sentimientos](https://colab.research.google.com/drive/1CzrbJVRNDXXsiP752i0PNpjq1TyAy-Tm#scrollTo=bkxtqgMBLG5q)

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

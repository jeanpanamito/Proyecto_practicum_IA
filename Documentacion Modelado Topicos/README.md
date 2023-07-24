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
En una nueva colección llamada ecuadorTweets, se cargaron los datos preprocesados para acceder a ellos con una mayor eficiencia.
Dentro de la colección trabajaremos con el atributo clean_text el cual tiene la información de cada tweet.

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
## Exploración de parámetros para modelo LDA
Es importante realizar todas las pruebas necesarias para identificar los mejores parámetros con el fin de definirlos en nuestro modelo.

En nuestro modelo de exploración, tomamos como prueba **30 tópicos** y asignamos que el valor de Alpha este comprendido en el siguiente rango **list(np.arange(0.01, 1, 0.3))** donde puede ser _symmetric_ o _asymmetric_, de la misma forma a Beta le otorgamos el rango de **list(np.arange(0.01, 1, 0.3))** con opción a ser _symmetric_. 

```python
import numpy as np
import tqdm

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 31
step_size = 1
topics_range = range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
               corpus]

corpus_title = ['100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))

    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title)
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('./results/lda_tuning_results.csv', index=False)
    pbar.close()
```

La información obtenida de este modelo de exploración se la guarda en un ![CSV](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/Archivos/lda_tuning_results.csv)

Déspues de leer y analizar su contenido, se identifica los valores óptimos para Alpha y Beta donde la coherencia entre tópicos sea la más alta.

Para este proyecto, encontramos que la mayor coherencia se la obtenia cuando Alpha y Beta estaban en **0.01**.

_Nota: Es importante recalcar que se hizo un experimento con los valores por defecto de Alpha y Beta, pero NO se obtuvo los resultado esperados._

Con estos parámetros ya definidos, realizamos un modelo LDA para conocer la cantidad de tópicos recomendada donde no haya intersección entre tópicos

```python
from gensim.models import CoherenceModel
coherence = []
for k in range(2,31):
    print('Round: '+str(k))
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=k, id2word = id2word, passes=20, iterations=20, chunksize = 25,random_state=10, eval_every = None,alpha=0.01,
                                           eta=0.01)

    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts= tweets,dictionary= id2word, coherence='c_v')

    coherence.append((k,cm.get_coherence()))
```
Imprimimos el número de tópicos junto con el nivel de coherencia y lo guardamos en un CSV
```python
listacv=[]
for i in range(0,29):
  listacv.append(coherence[i][1])
  print(listacv)
  print(len(listacv))
```
Con el uso de Matplotlib generamos una gráfica donde podemos visualizar cual es la cantidad recomendada de tópicos.
```python
import matplotlib.pyplot as plt
x= range(2,31)
plt.plot(x, listacv)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show
```
![Coherencia](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/pictures/Coherencia.png)

Con la gráfica nos damos cuenta que su pico es 2, pero no podemos tomar una cantidad tan mínima para nuestro modelado, es por eso que el valor que consideramos que aportaría un poco más sería el de **6 tópicos**.

Por lo pronto, hemos los valores definidos son:
-Número de Tópicos: 6
-Alpha: 0.01
-Beta: 0.01

Procedemos a hacer nuestro modelo LDA final.
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
Imprimimos los temas de los 6 tópicos
```python
from pprint import pprint

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```
Para la visualización de la gráfica del modeloLDA final, hacemos uso de 
```python
!pip install pyLDAvis
```
_Nota: Dentro del entorno virtual Google colab, despues de instalar el módulo pyLDAvis, debemos volver al instalar las versiones correctas de numpy y pandas, ya que el módulo las desintala e instala otras versiones, las cuales no permiten que el código funcione._
```python
pip install pandas==1.5.3 numpy==1.22.4
```
El código que genera la gráfica es el siguiente:

```python
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis_data_filepath = os.path.join('./results/ldavis_tuned_'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_tuned_'+ str(num_topics) +'.html')

LDAvis_prepared
```
![Topicos](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/pictures/Topicos.png)

Esa sería la gráfica resultante de nuestro modelo.

El enlace para observar el notebook de Google Colab es el siguiente https://colab.research.google.com/drive/1_oGORY3LoNTJTo5U3VP_S5SD4jj3rJHU?usp=sharing 


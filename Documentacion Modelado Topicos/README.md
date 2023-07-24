# MODELADO DE TOPICOS
### Importacion de librerias / módulos
En este proyecto, se han importado diversas librerías y módulos especializados para potenciar el procesamiento del lenguaje natural y el modelado de tópicos. Lo que nos permite tener un análisis exhaustivo de los datos textuales y la extracción de temas relevantes de manera eficiente.

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

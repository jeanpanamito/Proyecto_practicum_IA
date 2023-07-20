# Imports
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# nltk.download('punkt')
# nltk.download('stopwords')

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


# Funcion Preprocesado

def preprocesadoV2(text):
    text = re.sub(r'@(\w+)([A-Z][a-z]+)', r'\1 \2', text)
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remover enlaces
    text = re.sub(r'[^\w\s,áéíóúÁÉÍÓÚ]', '', text)  # Remover caracteres especiales (excepto letras con acento)
    text = re.sub(r'\s+', ' ', text)  # Remover espacios duplicados
    text = re.sub(r'[^\w\s]', '', text)  # Remover puntuacion
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remover emojis
    text = text.lower()
    return text.strip()


def preprocesadoV1(text):
    text.lower()
    text = re.sub(r'\d+', '', text)


def remove_url(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


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


def tokenize(text):
    # Tokenizar el texto en palabras individuales
    tokens = word_tokenize(text, language='spanish')

    # Eliminar palabras vacías (stop words)
    stop_words = set(stopwords.words('spanish'))
    stop_words.update("rt")
    tokens = [word for word in tokens if word not in stop_words]

    # Unir las palabras nuevamente en un solo texto
    # preprocessed_text = ' '.join(tokens)

    # return preprocessed_text
    return tokens


# Diccionario de provincias del Ecuador
provincias_dict = {
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
regex_pattern = re.compile(r'\bmis' + '|'.join(provincias_dict.keys()) + r'sing_value\b', re.IGNORECASE)

# Base de Datos y Colección
mongo_db = 'Preprocessing'
mongo_collection = 'tweets'
db = client[mongo_db]

# Consulta de los datos
datos = db.tweets.find()

# Obtencion de tweets solo de Ecuador sin retweets
ecuador_tweets = []
tweet_texts = set()

for dato in datos:
    tweet_text = dato['full_text']

    if tweet_text not in tweet_texts:
        tweet_texts.add(tweet_text)
        ecuador_tweets.append(dato)

# Preprocesado de los tweets de Ecuador
for ecutweet in ecuador_tweets:
    full_text = ecutweet['full_text']
    id = ecutweet['id']
    print(f'------------Full Text - {id} ------------\n '
          f'{full_text}')
    # Convertir a minusculas, numeros, rt y puntuacion
    n1 = (preprocess
          (remove_punctuation
           (remove_url
            (remove_numbers
             (remove_rt
              (text_lowercase(full_text)))))))
    n2 = (preprocesadoPrueba(full_text))
    wordsV1 = tokenize(n1)
    wordsV2 = tokenize(n2)
    print(f'---------------Clean Text V1----------------\n '
          f'{wordsV1}')

    print(f'---------------Clean Text V2----------------\n '
          f'{wordsV2}')

print(len(ecuador_tweets))

# spell = SpellChecker(language='es')
# for word in words:
# Check if the word is misspelled
# if not spell.correction(word) == word:
# Get the most likely correct spelling
# correction = spell.correction(word)
# print(f"The word '{word}' is misspelled. Did you mean '{correction}'?")


#if re.search(regex_pattern, dato['full_text']) \
        #        or re.search(regex_pattern, dato['user']['location']) \
        #        or re.search(regex_pattern, ' '.join(dato['hashtags'])):
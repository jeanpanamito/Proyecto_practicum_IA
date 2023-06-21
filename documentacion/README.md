# Proyecto_practicum_IA
Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional
# Informe de actividades

## Introducción:

Durante este proyecto, hemos explorado el tema del análisis de sentimientos en tweets en español. En particular, se hizo referencia a un trabajo titulado "Análisis de Sentimientos de Tweets en Español Basado en Técnicas de Aprendizaje Supervisado" realizado como Trabajo Fin de Grado durante el curso 2020-2021 por Juan Antonio Carrión García, Rodrigo Fernández Ambrona, Cristina Molina Gerbolés y Patricia Motoso González, bajo la dirección de Francisco Javier Crespo Yáñez y Luis Javier García Villalba, en el Departamento de Sistemas Informáticos y Computación de la Facultad de Informática de la Universidad Complutense de Madrid.

El trabajo se centró en el análisis de sentimientos de tweets en español utilizando técnicas de aprendizaje supervisado. Se utilizaron las siguientes herramientas y librerías:

| Nombre        | Tipo         | Uso           | Referencia                           |
|---------------|--------------|---------------|--------------------------------------|
| TextBlob      | Librería     | Libre         | Gujjar & HR 2021                     |
| NLTK          | Kit de herramientas | Libre   | Loper & Bird 2002                     |
| OpenNLP       | Librería     | Libre         | Kottmann et al. 2011                  |
| VADER         | Léxico       | Libre         | Borg & Boldt 2020                     |
| GATE          | Kit de herramientas | Libre   | Cunningham 2002                       |

Estas herramientas y librerías fueron utilizadas para el análisis de sentimientos de tweets en español en el trabajo mencionado. Es importante destacar que nuestro proyecto ha utilizado algunas de estas herramientas en nuestro análisis de sentimientos de tweets.

## Objetivo:

El objetivo de este proyecto fue aplicar técnicas de aprendizaje supervisado para realizar un análisis de sentimientos en tweets escritos en español. El análisis de sentimientos se refiere a la tarea de determinar la polaridad emocional de un texto, es decir, si el texto expresa emociones positivas, negativas o neutrales.

## Herramientas Utilizadas:

Durante este proyecto, mencionaremos varias herramientas y bibliotecas que se utilizan comúnmente en el análisis de sentimientos en el procesamiento de lenguaje natural. A continuación, se detallan algunas de ellas:

- **Natural Language Toolkit (NLTK):** Es una biblioteca de procesamiento de lenguaje natural ampliamente utilizada en Python. Proporciona herramientas para el procesamiento de texto, tokenización, lematización, análisis gramatical, etc.

- **spaCy:** Otra biblioteca popular de procesamiento de lenguaje natural en Python. Ofrece funcionalidades avanzadas como el análisis sintáctico y el reconocimiento de entidades nombradas.

- **TextBlob:** Es una biblioteca sencilla y fácil de usar para el procesamiento de lenguaje natural en Python. Proporciona herramientas para el análisis de sentimientos, etiquetado gramatical, traducción, entre otros.

- **TensorFlow** y **PyTorch:** Son bibliotecas populares de aprendizaje de máquina y deep learning en Python. Se utilizan para construir y entrenar modelos de análisis de sentimientos más complejos, como las redes neuronales.

Además de estas herramientas, también se mencionó la utilización de diccionarios de sentimientos, como SentiWordNet, que contienen palabras y su polaridad emocional asociada. Estos diccionarios son utilizados para asignar polaridades a las palabras en el análisis de sentimientos.

## Características de los Modelos Evaluados:

Durante este proyecto, se han revisado diferentes modelos de análisis de sentimientos, incluyendo el modelo VADER (Valence Aware Dictionary and sEntiment Reasoner). El modelo VADER es un enfoque basado en reglas que utiliza diccionarios de palabras y reglas gramaticales para determinar la polaridad de un texto.

En cuanto a los modelos de la biblioteca NLTK, esta ofrece diferentes enfoques para el análisis de sentimientos, como el clasificador Naive Bayes y el clasificador de Máxima Entropía. Estos modelos pueden ser entrenados utilizando conjuntos de datos etiquetados con polaridades emocionales.

Además, se realizo la utilización de la API de RoBERTa, un modelo basado en transformers para el procesamiento de lenguaje natural. RoBERTa es un modelo pre-entrenado que puede ser afinado (fine-tuned) en tareas específicas, como el análisis de sentimientos, utilizando conjuntos de datos adecuados.

### Niveles de Análisis de Sentimientos

Vamos a tener tres niveles distintos para llevar a cabo el análisis de sentimientos de este proyecto:

- **Análisis a nivel de documento:** Nivel en el cual se va a analizar qué tipo de sentimiento se encuentra en un documento, pero a través de una visión global como un todo. Se va a clasificar en positivo, negativo y neutro.

- **Análisis a nivel de oración:** Nivel en el que se divide un documento en oraciones individuales y con esto podremos extraer la opinión de cada una de ellas dividiéndolas en opiniones positivas, negativas y neutras.

- **Análisis a nivel de aspecto y entidad:** Nivel que se a analizar con mayor detalle. Consiste en una entidad más pequeña que va a estar formada por distintos elementos, y se expresará una opinión de cada uno de ellos.

## Precisión de los Modelos:

La precisión de los modelos de análisis de sentimientos puede variar según diversos factores, como la calidad del conjunto de datos utilizado para el entrenamiento y la evaluación, la representación de las características utilizadas por el modelo, la técnica de aprendizaje supervisado aplicada y la complejidad de la tarea de análisis de sentimientos.
Una vez hecho el preprocesado del texto se necesita un sistema de ponderación numérica. Los dos sistemas de ponderación más relevantes son:

- **TF (Frecuencia de Término):** Muestra qué tan frecuente es un término en un documento.

$$
tf(t, d) = \frac{f(t, d)}{\max f(t, d) : t \in d}
$$

- **TF-IDF (Frecuencia de Término-Inversa de Frecuencia de Documento):** Se compone de dos partes. En primer lugar, se utiliza TF para mostrar qué tan frecuente es un término en un documento. Por otro lado, se encuentra IDF, que se refiere a la frecuencia inversa de un término en los documentos. Esto se utiliza para reducir el peso de un término. El cálculo de TF-IDF se realiza mediante la siguiente fórmula:


La fórmula de IDF (Frecuencia de Término-Inversa de Frecuencia de Documento) se representa de la siguiente manera:

$$
idf(t, D) = \frac{\log |D|}{|d \in D : t \in d|}
$$

La fórmula de TF-IDF (Frecuencia de Término-Inversa de Frecuencia de Documento) se representa de la siguiente manera:

$$
tf-idf(t, d, D) = tf(t, d) \cdot idf(t, D)
$$


#### Modelo-Roberta 
![Tweet Sentiment- RoBERTa](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/pictures/roberta.jpg)
#### Modelo-Vader
![Tweet Sentiment- RoBERTa](https://github.com/jeanpanamito/Proyecto_practicum_IA/blob/main/pictures/vader.jpg)

## Conclusiones:

En este proyecto, hemos explorado el análisis de sentimientos de tweets en español y hemos hecho referencia a un trabajo específico realizado en la Universidad Complutense de Madrid. Aunque no se pas herramientas utilizadas en ese trabajo, se destacaron bibliotecas populares de procesamiento de lenguaje natural (NLP) y aprendizaje de máquina, como NLTK, spaCy, TextBlob, scikit-learn, TensorFlow y PyTorch. Además, se mencionó la utilización de diccionarios de sentimientos como SentiWordNet.







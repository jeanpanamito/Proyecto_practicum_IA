# Proyecto_practicum_IA
Aplicación de Técnicas de Inteligencia Artificial en Información Relacionada a la Seguridad Nacional
# Informe de actividades

## Introducción:

Durante este proyecto, hemos explorado el tema del análisis de sentimientos en tweets en español. En particular, se hizo referencia a un trabajo titulado "Análisis de Sentimientos de Tweets en Español Basado en Técnicas de Aprendizaje Supervisado" realizado como Trabajo Fin de Grado durante el curso 2020-2021 por Juan Antonio Carrión García, Rodrigo Fernández Ambrona, Cristina Molina Gerbolés y Patricia Motoso González, bajo la dirección de Francisco Javier Crespo Yáñez y Luis Javier García Villalba, en el Departamento de Sistemas Informáticos y Computación de la Facultad de Informática de la Universidad Complutense de Madrid.

El trabajo se centró en el análisis de sentimientos de tweets en español utilizando técnicas de aprendizaje supervisado. Se utilizaron las siguientes herramientas y librerías:

| Nombre        | Tipo         | Uso           | Referencia                           |
|---------------|--------------|---------------|--------------------------------------|
| FreeLing      | Librería     | Libre         | Carreras et al. 2004                  |
| TextBlob      | Librería     | Libre         | Gujjar & HR 2021                     |
| NLTK          | Kit de herramientas | Libre   | Loper & Bird 2002                     |
| OpenNLP       | Librería     | Libre         | Kottmann et al. 2011                  |
| Quanteda      | Librería     | Libre         | Benoit et al. 2018                    |
| SentiWordNet  | Léxico       | Libre         | Esuli & Sebastiani 2006               |
| Bing Liu      | Léxico       | Libre         | Liu 2012                              |
| Sentiment 140 | Léxico       | Libre         | Go et al. 2009                        |
| AFINN         | Léxico       | Libre         | Arup Nielsen 2011                     |
| SenticNet     | Léxico       | Libre         | Cambria et al. 2014                   |
| VADER         | Léxico       | Libre         | Borg & Boldt 2020                     |
| CoreNLP       | Conjunto de herramientas | Libre | Manning et al. 2014              |
| GATE          | Kit de herramientas | Libre   | Cunningham 2002                       |
| LingPipe      | Kit de herramientas | Libre   | Carpenter 2007                        |
| MALLET        | Kit de herramientas | Libre   | McCallum 2002                          |
| OpinionFinder | Conjunto de herramientas | Libre | Wilson et al. 2005                |
| LIWC          | Conjunto de herramientas | Cuota | Pennebaker et al. 2001           |
| AutoML        | Conjunto de herramientas | Demo/Cuota | Zeng & Zhang 2020              |
| Monkey Learn  | Conjunto de herramientas | Demo/Cuota | Monkey-Learn 2020               |
| Hootsuite     | Herramienta  | Demo/Cuota    | HootSuite, WE ARE SOCIAL Y 2020      |

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

## Precisión de los Modelos:

La precisión de los modelos de análisis de sentimientos puede variar según diversos factores, como la calidad del conjunto de datos utilizado para el entrenamiento y la evaluación, la representación de las características utilizadas por el modelo, la técnica de aprendizaje supervisado aplicada y la complejidad de la tarea de análisis de sentimientos.

![Tweet Sentiment- RoBERTa](ruta/de/la/imagen.jpg)


## Conclusiones:

En este proyecto, hemos explorado el análisis de sentimientos de tweets en español y hemos hecho referencia a un trabajo específico realizado en la Universidad Complutense de Madrid. Aunque no se pas herramientas utilizadas en ese trabajo, se destacaron bibliotecas populares de procesamiento de lenguaje natural (NLP) y aprendizaje de máquina, como NLTK, spaCy, TextBlob, scikit-learn, TensorFlow y PyTorch. Además, se mencionó la utilización de diccionarios de sentimientos como SentiWordNet.


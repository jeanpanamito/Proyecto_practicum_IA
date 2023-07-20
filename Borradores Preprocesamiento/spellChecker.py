from spellchecker import SpellChecker
import nltk

# Create a SpellChecker object for Spanish language
spell = SpellChecker(language='es')

# Tokenize the tweet into words
def tokenize_tweet(tweet):
    tokens = nltk.word_tokenize(tweet)
    return tokens

# Example tweet
tweet = "Hola amigos! Espero que estén teniendo un buen día. ¡Nos vemos luego!"

# Tokenize the tweet
words = tokenize_tweet(tweet)

# Check spelling of each word
for word in words:
    # Check if the word is misspelled
    if not spell.correction(word) == word:
        # Get the most likely correct spelling
        correction = spell.correction(word)
        print(f"The word '{word}' is misspelled. Did you mean '{correction}'?")

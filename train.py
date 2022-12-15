import spacy
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from data import intents 

stemmer = PorterStemmer()

nltk.download('punkt') # Can be commented out once it is downloaded

lemmatizer = nltk.WordNetLemmatizer()


questions = []
stemmed_questions = []
tags = []

tokens = set()

tags_and_replies = {}

tag_idx = 0
tag_to_idx = {}

for intent in intents:
    for q in intent["questions"]:
        questions.append(q)
        words = word_tokenize(q)
        stemmed_questions.append([])
        for word in words:
            tokens.add(stemmer.stem(word))
            stemmed_questions[-1].append(stemmer.stem(word))
        if stemmed_questions[-1]:
            stemmed_questions[-1] = " ".join(stemmed_questions[-1])

        tags.append(intent["tag"])
        tags_and_replies[intent["tag"]] = intent["answers"]
    tag_to_idx[intent["tag"]] = tag_idx
    tag_idx += 1



count_vectorizer = CountVectorizer(ngram_range=(1,2))

X_train = count_vectorizer.fit_transform(stemmed_questions)

clf = MultinomialNB()
clf.fit(X_train, tags)

from joblib import dump, load
dump({"clf":clf,"tags_and_replies":tags_and_replies, "vectorizer":count_vectorizer}, 'model.joblib') 

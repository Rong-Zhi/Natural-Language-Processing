# Words as Featrues
# In this tutorial, we're going to be building off
# compiling feature lists of words
# from positive reviews and words from the negative reviews
# to hopefully see trends in specific types of words
# in positive or negative reviews.


####################
### Movie review ###
####################

import nltk
import random
from nltk.corpus import movie_reviews

# a list of tuples --- features
documents = [(list(movie_reviews.words(fileid)),category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# documents = []
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid)), category)

random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) # the words here may be useless

word_features = list(all_words.keys())[:3000] # we need more words that we commonly use

def find_features(document):
    words = set(document)   # delete duplicates
    features = {}   # dictionary
    for w in word_features:
        features[w] = (w in words)
    return features

# we use a negative words repository to check the top 3000 words
# false: not negative, ture: is negative
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev, category) in documents]
# Text Classification

# Maybe we're trying to classify text as about politics
# or the military. Maybe we're trying to classify it by
# the gender of the author who wrote it.
# A fairly popular text classification task is to
#  identify a body of text as either spam or not spam,
#  for things like email filters.

# In our case, we're going to try to
# create a sentiment analysis algorithm.

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

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

print(all_words["stupid"])

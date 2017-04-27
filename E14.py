# Save Classifier with Pickle
# save the trained module

# import pickle

####################
### Movie review ###
####################

import pickle
import nltk
import random
from nltk.corpus import movie_reviews

#   a list of tuples --- features
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

#   we use a negative words repository to check the top 3000 words
#   false: not negative, ture: is negative
# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#   posterior = prior occurences * liklihood / evidence
# classifier = nltk.NaiveBayesClassifier.train(training_set)
#  uncomment when you want to train the module

# then open the saved module here
classifier_f = open("naivebyes.pickle")
classifier = pickle.load(classifier_f)
classifier_f.close()

# * 100 to get percent
print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)
# the most valuable words are when it comes to positive or negative reviews
# given by ratio
classifier.show_most_informative_features(15)

# first save the module here
# save_classifier = open("naivebayes.pickle","wb") # wb--write in bytes
# pickle.dump(classifier, save_classifier)
# save_classifier.close()
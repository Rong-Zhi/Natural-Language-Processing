# Sentiment Analysis Module


##################
#### Twitter #####
##################

import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return  mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        # count how may that most popular words occurances
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

############# Access the files ###############
short_pos = open("positive.txt","r",errors='ignore').read()
short_neg = open("negative.txt","r",  errors='ignore').read()


all_words = []
documents = []


#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

#######
save_documents = open("documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()



all_words = nltk.FreqDist(all_words) # the words here may be useless

word_features = list(all_words.keys())[:5000] # we need more words that we commonly use

#######
save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}   # dictionary
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev),category) for (rev, category) in documents]

save_featuresets = open("featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print(len(featuresets))
#
training_set = featuresets[:10000]
testing_set = featuresets[10000:]


#   posterior = prior occurences * liklihood / evidence
classifier = nltk.NaiveBayesClassifier.train(training_set)
#  uncomment when you want to train the module

# * 100 to get percent
#   now we can change other classifier
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)
# the most valuable words are when it comes to positive or negative reviews
# given by ratio
classifier.show_most_informative_features(15)

#######
save_classifier = open("orignalnaivebayes5k.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#######
save_classifier = open("MultinomialNB5k.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

#######
save_classifier = open("BernoulliNB5k.pickle","wb")
pickle.dump(BNB_classifier,save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#######
save_classifier = open("LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

#######
save_classifier = open("SGDClassifier_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

#######
save_classifier = open("SVC_classifier5k.pickle","wb")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

#######
save_classifier = open("LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:",
#       (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
#
# #######
# save_classifier = open("NuSVC_classifier5k.pickle","wb")
# pickle.dump(NuSVC_classifier,save_classifier)
# save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  # NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier)

# print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
#
# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
#



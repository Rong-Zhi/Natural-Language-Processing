# WordNet

# WordNet is a lexical database for the English language,
# which was created by Princeton, and is part of the NLTK corpus.
#
# You can use WordNet alongside the NLTK module to find
# the meanings of words, synonyms, antonyms, and more.
# Let's cover some examples.


from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# print(syns)

print(syns[0].name()) # output: plan.n.01

print(syns[0].lemmas()[0].name()) # output: plan

print(syns[0].definition()) # output: a series of steps to be carried out or goals to be accomplished
                            # The output is the definition of program
print(syns[0].examples()) # given some example sentences

synonyms = []   # a word or phrase that means exactly or
                # nearly the same as another word or phrase
antonyms = []   # opposite meaning

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        xxx=l.name()
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms)) # create a set without duplicates
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")

print(w1.wup_similarity(w2)) # compare semantic similarity in percent
                            # output: 0.909

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))    # output: 0.69

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))    # output: 0.32
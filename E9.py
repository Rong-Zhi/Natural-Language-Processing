#Corpora dataset

import nltk
# print(nltk.__file__) # the path of nltk

#         str('/usr/share/nltk_data'),
#         str('/usr/local/share/nltk_data'),
#         str('/usr/lib/nltk_data'),
#         str('/usr/local/lib/nltk_data')
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)


print(tok[5:15])
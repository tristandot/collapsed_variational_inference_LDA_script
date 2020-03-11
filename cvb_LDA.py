#!/usr/bin/env python
# coding: utf-8

import sys
import io
import time
import nltk
import numpy as np
from itertools import groupby
from unidecode import unidecode
from nltk.corpus import stopwords

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from cvbLDA import cvbLDA
from text_process import text_to_indices

#Take into account the different arguments given by user
file_path = sys.argv[1]
language = sys.argv[2]
nb_topics = int(sys.argv[3])
nb_words_topic = int(sys.argv[4])

mode = 'true'
user_lda_paramters = False

if(len(sys.argv) == 6):
    mode = sys.argv[5]

elif(len(sys.argv) > 6):
    user_lda_paramters = True
    alpha = float(sys.argv[6])
    beta = float(sys.argv[7])
    nb_iter_max = int(sys.argv[8])
    tol = float(sys.argv[9])

#Loading data and processing it (lower, remove stopwords, remove punctuation), and transform it to SVMLight-style format
print("\n######## DATA LOADING AND PROCESSING ########")

f = io.open(file_path, 'r')

tree_repr_db = []
for line in f:
    tree_repr = unidecode(line.strip())
    if len(tree_repr) == 0:
        continue

    while tree_repr[-1] == ' ':
        tree_repr = tree_repr[:-1]

    tree_repr_db.append(nltk.word_tokenize(tree_repr))
    pass


nltk.download('stopwords')
if(language == 'french'):
    stop_words = nltk.corpus.stopwords.words('french')
    stop_words.extend(['est','chez','avant','apres','plus','moins','etre','tres','cela','ca','avoir','ete','si','je','a','le','la','les','il','elle','l','elles','ils','comme','cette','ce','cet'])

else:
    stop_words = nltk.corpus.stopwords.words('english')

corpus = tree_repr_db


#lower it and remove stop words
corpus=[[word.lower() for word in corpus[i] if word.isalpha()] for i in range(len(corpus))]

corpus=[[word for word in corpus[i] if word not in stop_words] for i in range(len(corpus))]

#put our input in a SVMLight-style format
[text_numbers,word_dict] = text_to_indices(corpus)
text_numbers=list(map(sorted, text_numbers))

words=[ sorted(list(set(text_numbers[j]))) for j in range(len(text_numbers))]
counts=[[len(list(group)) for key, group in groupby(text_numbers[i])] for i in range(len(text_numbers))]

#Dictionnary to retrieve words from their index
index_dict = {v: k for k, v in word_dict.iteritems()}

#Fix the vocab size and the nb of topics, for the collapsed variational LDA algorithm
vocab_size = len(index_dict)
print('\nVOCAB SIZE: ' + str(vocab_size) + ' WORDS')


#Collapsed vi LDA implementation
print("\n######## COLLAPSED VARIATIONAL INFERENCE LDA ########")

if(user_lda_paramters == False):
    #Parameters alpha and beta taken as in the article
    alpha = .1 * np.ones((1,nb_topics))
    beta = .1 * np.ones((nb_topics,vocab_size))

    nb_iter_max = 100
    tol = .001

#If the user has specified some parameters, we fix them
else:
    alpha = alpha * np.ones((1,nb_topics))
    beta = beta * np.ones((nb_topics,vocab_size))


start_time = time.time()

(phi,theta,gamma) = cvbLDA(words,counts,alpha,beta,
                           maxiter=nb_iter_max,verbose=0,convtol=tol)

print('\nCollapsed variational inference LDA exec time: ' + str(time.time() - start_time) + 's')

#print('Theta, p(z|d)')
#print(str(theta))

#print('Phi, p(w|z)')
#print(str(phi))

topic=[]

#Print the nb_words_topic main words of each topic
print("\nTopics found via collapsed variational inference LDA: ")

for j in range(nb_topics):
    topic.append(np.array(phi)[j].argsort()[-nb_words_topic:][::-1])
    probas=np.sort(np.array(phi)[j])[::-1]

    print('\nTopic '+str(j+1)+ ': ' + str(nb_words_topic) + ' most important words, with p(w|z):')
    for i in range(len(topic[j])):
        print(index_dict[topic[j][i]]+', ' + str(round(probas[i]*100,7))+'%')

#Link each document to its main topic, thanks to a dictionnary
doc_topic={}
for j in range(len(corpus)):
    doc_topic[j] = [list(np.array(theta)[j].argsort()[-1:][::-1])[0],np.sort(np.array(theta))[j][::-1][0]]

#print(doc_topic)

#If user has specified comparison, comparison with the LDA Sklearn implementation performances
if(mode=='true'):

    print("\n######## SKLEARN LDA COMPARISON ########")

    f = io.open(file_path, 'r')
    test = []
    for line in f:
        tree_repr = unidecode(line.strip())
        if len(tree_repr) == 0:
            continue

        while tree_repr[-1] == ' ':
            tree_repr = tree_repr[:-1]

        test.append(tree_repr)
        pass

    vectorizer = CountVectorizer(stop_words=stop_words)
    data = vectorizer.fit_transform(test)

    start_time = time.time()

    lda = LatentDirichletAllocation(n_components=nb_topics,   random_state=0)
    lda.fit(data)

    print('\nSklearn LDA exec time: ' + str(time.time() - start_time) + 's')

    #Print the nb_words_topic main words of each topic, for the Sklearn LDA implementation
    print("\nTopics found via Sklearn LDA: ")
    words = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]):
        print("\nTopic " + str(topic_idx+1) +': ' + str(nb_words_topic) + ' most important words, with p(w|z):')
        toto = topic.argsort()[:-nb_words_topic - 1:-1]
        probas=np.sort(np.array(topic))[::-1]
        for i in range(len(toto)):
            print(words[toto[i]] +', ' + str(round(probas[i]*100,7))+'%')

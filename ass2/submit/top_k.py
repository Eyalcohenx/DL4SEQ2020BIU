import sys

from numpy import dot
from numpy.linalg import norm
import numpy as np

vocab = open("vocab.txt").read().splitlines()
wordVectors = [np.asarray(line.split()).astype(np.float) for line in open("wordVectors.txt").readlines()]


def cos(x, y):
    return dot(x, y) / (norm(x) * norm(y))


def most_similar(word, k):
    word_vec = wordVectors[vocab.index(word)]
    similarity = np.asarray([cos(word_vec, vec) for vec in wordVectors])
    indexes = similarity.argsort()[-(k + 1):]
    for index in indexes:
        if vocab[index] != word:
            print("word:", vocab[index], ", distance:", cos(word_vec, wordVectors[index]))


# gets the word from the command line
most_similar(sys.argv[1], 5)


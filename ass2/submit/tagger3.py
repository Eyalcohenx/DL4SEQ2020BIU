import sys

import dynet_config
dynet_config.set_gpu()
dynet_config.set(autobatch=True)
import random
import re
import numpy as np
import dynet as dy
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

START = '<s>'
END = '</s>'
UNK = "UUUNKKK"

from contextlib import contextmanager
from collections import defaultdict
from time import time

times_dict = defaultdict(float)


@contextmanager
def timethis(label):
    t0 = time()
    yield
    times_dict[label] += time() - t0


digit_patterns = [re.compile('[0-9]+')]


def replace_digits(pair):
    for pattern in digit_patterns:
        if pattern.match(pair[0]):
            pair[0] = re.sub('[0-9]', 'DG', pair[0])
    return pair


def prefixes_suffixes(words):
    prefixes_suffixes = []
    for word in words:
        if len(word) >= 3:
            prefixes_suffixes.append("pref_" + word[:3])
            prefixes_suffixes.append("suff_" + word[-3:])
    prefixes_suffixes.append("pref_SHORT")
    prefixes_suffixes.append("suff_SHORT")
    return list(set(prefixes_suffixes))


def process_line(line):
    return [replace_digits([tok, tag]) for tok, tag in [pair.split() for pair in line.split("\n")]]


def read_file(file):
    with open(file, 'r') as file:
        data = file.read().replace('\n\n', '_')
        return [process_line(line) for line in data.split("_") if not len(line) == 0]


def to_triplet_suff_pre_dev(word):
    if len(word) >= 3:
        if word in vocabulary:
            word1 = word
        else:
            word1 = UNK
        if "pref_" + word[:3] in vocabulary:
            pref1 = "pref_" + word[:3]
        else:
            pref1 = "pref_" + UNK[:3]
        if "suff_" + word[-3:] in vocabulary:
            suff1 = "suff_" + word[-3:]
        else:
            suff1 = "suff_" + UNK[-3:]
        return np.array([word1, pref1, suff1])
    else:
        if word in vocabulary:
            word2 = word
        else:
            word2 = UNK
        return np.array([word2, "pref_SHORT", "suff_SHORT"])


def to_triplet_suff_pre_train(word):
    if len(word) >= 3:
        return np.array([word, "pref_" + word[:3], "suff_" + word[-3:]])
    else:
        return np.array([word, "pref_SHORT", "suff_SHORT"])


def update_vocab(sents, vocab, vectors):
    w2i = {w:i for i, w in enumerate(vocab)}
    vocab_set = set(vocab)
    org_vocab_set = set(vocab)
    mean_vec = np.mean(vectors, axis=0)
    max_mean = np.max(mean_vec)
    min_mean = np.min(mean_vec)
    size_mean = mean_vec.size
    mean_vec = np.mean(vectors, axis=0)
    for sent in sents:
        for pair in sent:
            if pair[0] not in vocab_set:
                if pair[0].lower() in org_vocab_set:
                    vocab.append(pair[0])
                    vocab_set.add(pair[0])
                    vectors.append(vectors[w2i[pair[0].lower()]])
                else:
                    vocab.append(pair[0])
                    vocab_set.add(pair[0])
                    vectors.append(np.random.uniform(low=min_mean, high=max_mean, size=size_mean))




def create_features(sents, words_encoder, tags_encoder, to_triplet_suff_pre):
    start_idx, end_idx = words_encoder.transform([START, END])
    words = [to_triplet_suff_pre(pair[0]) for sent in sents
             for pair in [(START, 0), (START, 0)] + sent + [(END, 0), (END, 0)]] + [to_triplet_suff_pre(END)]
    tags = [pair[1] for sent in sents for pair in sent]
    words = np.array(words)  # size 1053467
    words = np.concatenate(words).ravel()
    word_encs = words_encoder.transform(words)
    word_encs = np.reshape(word_encs, (int(len(word_encs) / 3), 3))
    window_encs = np.stack([word_encs[i:i - 5] for i in range(5)], axis=-2)
    window_encs = [element for element in window_encs if element[2][0] != start_idx and element[2][0] != end_idx]
    tag_encs = tags_encoder.transform(tags)
    return list(zip(window_encs, tag_encs))


# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the parameter collection.
    def __init__(self, pc, DIM, VOCAB_SIZE, out_dim, mid_dim, word_vectors=None):
        self.pW = pc.add_parameters((mid_dim, DIM * 5))
        self.pB_1 = pc.add_parameters(mid_dim)
        self.pV = pc.add_parameters((out_dim, mid_dim))
        self.pB_2 = pc.add_parameters(out_dim)
        self.E = pc.add_lookup_parameters((VOCAB_SIZE, DIM))
        if word_vectors is not None:
            self.E.init_from_array(word_vectors)

    # the __call__ method applies the network to an input
    def __call__(self, inputs):
        lookup = self.E
        emb_vectors = [dy.esum([lookup[vecs[0]], lookup[vecs[1]], lookup[vecs[2]]]) for vecs in inputs]
        net_input = dy.concatenate(emb_vectors)
        net_output = dy.softmax(self.pV * (dy.tanh((self.pW * net_input) + self.pB_1)) + self.pB_2)
        return net_output

    def create_network_return_loss(self, inputs, expected_output):
        out = self(inputs)
        loss = -dy.log(dy.pick(out, expected_output))
        return loss

    def create_network_return_best(self, inputs):
        dy.renew_cg()
        out = self(inputs)
        return dy.np.argmax(out.npvalue())


if __name__ == '__main__':
    trainer_state = sys.argv[1]

    corpus_file_location = ""
    dev_file_location = ""
    if trainer_state == "pos":
        corpus_file_location = 'pos/train'
        dev_file_location = 'pos/dev'
    else:
        corpus_file_location = 'ner/train'
        dev_file_location = 'ner/dev'

    # reading curpus_file
    train_sents = read_file(corpus_file_location)
    train_data = [pair for sent in train_sents for pair in sent]

    vocab = open("vocab.txt").read().splitlines()
    wordVectors_ = [np.asarray(line.split()).astype(np.float) for line in open("wordVectors.txt").readlines()]
    update_vocab(train_sents, vocab, wordVectors_)

    # counting number of available tags
    tags_encoder = preprocessing.LabelEncoder()
    tags_encoder.fit([pair[1] for pair in train_data])

    # counting number of available tags and words
    words_encoder = preprocessing.LabelEncoder()
    prefsuff = prefixes_suffixes(vocab)
    vocab += prefsuff
    words_encoder.fit(vocab)
    mean_vec = np.mean(wordVectors_, axis=0)
    pref_suff_vecs = np.random.uniform(low=np.min(mean_vec), high=np.max(mean_vec), size=(len(prefsuff), len(mean_vec)))

    for vec in pref_suff_vecs:
        wordVectors_ += [vec]

    # reset the global cg
    dy.renew_cg()

    # create parameter collection
    m = dy.ParameterCollection()

    # constants
    DIM = 50
    VOCAB_SIZE = len(words_encoder.classes_)
    out_dim = len(tags_encoder.classes_)
    mid_dim = int((DIM * 5 + out_dim) / 2)
    alpha = 0.5
    epochs = 15
    batch_size = 256
    rounds_per_epoch = 4096

    w2i = {w: i for i, w in enumerate(vocab)}
    wordVectors = np.asarray([wordVectors_[w2i[w]] for w in words_encoder.inverse_transform(range(VOCAB_SIZE))])

    # create network
    network = OurNetwork(m, DIM=DIM, VOCAB_SIZE=VOCAB_SIZE, out_dim=out_dim, mid_dim=mid_dim, word_vectors=wordVectors / 30)

    # create trainer
    trainer = dy.SimpleSGDTrainer(m)
    trainer.learning_rate = alpha

    vocabulary = words_encoder.classes_
    dev_sents = read_file(dev_file_location)

    # creating feature data
    train_encoded = create_features(train_sents, words_encoder, tags_encoder, to_triplet_suff_pre_train)
    test_encoded = create_features(dev_sents, words_encoder, tags_encoder, to_triplet_suff_pre_dev)

    total_loss = 0
    seen_instances = 0

    # train network
    tag_encs = np.asarray([x[1] for x in train_encoded])
    for epoch in range(epochs):
        if epoch % 5 == 0 and epoch != 0:
            trainer.learning_rate = trainer.learning_rate / 2
        print("starting epoch: ", epoch)
        dy.renew_cg()
        window_encs_copy = np.asarray([x[0] for x in train_encoded])
        tag_encs_copy = np.asarray([x[1] for x in train_encoded])
        permute_idx = np.random.permutation(len(tag_encs))
        window_encs_copy, tag_encs_copy = window_encs_copy[permute_idx], tag_encs_copy[permute_idx]
        total_loss = 0
        seen_instances = 0
        n_batches = len(tag_encs) // batch_size
        for i in range(n_batches):
            dy.renew_cg()
            batch_idx = slice(batch_size * i, batch_size * (i + 1))
            x_sample = window_encs_copy[batch_idx]
            y_sample = tag_encs_copy[batch_idx]
            losses = []
            for inp, lbl in zip(x_sample, y_sample):
                 loss = network.create_network_return_loss(inp, lbl)
                 seen_instances += 1
                 losses.append(loss)
            batch_loss = dy.esum(losses) / batch_size
            if i % 500 == 0 and i != 0:
                print('loss:', batch_loss.npvalue()[0])  # this calls forward on the batch
            batch_loss.backward()
            trainer.update()

        if True:
            with timethis('validation'):
                # checking accuracy on dev set
                y_predicted = [network.create_network_return_best(x[0]) for x in test_encoded]
                y_true = [y[1] for y in test_encoded]

                if trainer_state == "pos":
                    # POS acc
                    acc = accuracy_score(y_true, y_predicted, normalize=True, sample_weight=None)
                else:
                    # NER acc
                    y_predicted = tags_encoder.inverse_transform(y_predicted)
                    y_true = tags_encoder.inverse_transform(y_true)

                    counter = 0
                    currect_counter = 0

                    for y, y_tag in zip(y_true, y_predicted):
                        if y == 'O' and y_tag == 'O':
                            none = 1
                        else:
                            if y == y_tag:
                                currect_counter += 1
                            counter += 1
                    try:
                        acc = currect_counter / counter
                    except ZeroDivisionError:
                        acc = 0.0

                print("Accuracy:", acc)
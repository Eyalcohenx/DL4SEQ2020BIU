# WITHOUT BATCHING
import dynet_config
dynet_config.set_gpu()
dynet_config.set(autobatch=True)
import random
import sys
import numpy as np
import dynet as dy
from sklearn import preprocessing
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, accuracy_score

START = '<START>'
END = '<END>'
UNK = '<UNK>'


class HasNextIterator:
    def __init__(self, it):
        self._it = iter(it)
        self._next = None

    def __iter__(self):
        return self

    def has_next(self):
        if self._next:
            return True
        try:
            self._next = next(self._it)
            return True
        except StopIteration:
            return False

    def next(self):
        if self._next:
            ret = self._next
            self._next = None
            return ret
        elif self.has_next():
            return self.next()
        else:
            raise StopIteration()


def process_line(line):
    return [pair.split() for pair in line.split("\n")]


_i_ = 0
max_n = 100


def read_file(file):
    with open(file, 'r') as file:
        data = file.read().replace('\n\n', '_')
        return [process_line(line) for line in data.split("_") if not len(line) == 0]


def create_feature_vecs(train_sents, words_encoder, tags_encoder):
    feature_vecs = []

    for sent in train_sents:
        it = HasNextIterator(sent)
        if it.has_next():
            it.next()

        if it.has_next():
            word_plus_1 = it.next()[0]
        else:
            word_plus_1 = END

        if it.has_next():
            word_plus_2 = it.next()[0]
        else:
            word_plus_2 = END

        word_minus_2, word_minus_1 = START, START

        for tok, tag in sent:
            window = [word_minus_2, word_minus_1, tok, word_plus_1, word_plus_2]
            try:
                feature_vecs.append([words_encoder.transform(window), tags_encoder.transform([tag])[0]])
            except ValueError:
                # handeling unknown words
                window_vec = []
                for word in window:
                    if word not in words_encoder.classes_:
                        window_vec.append(words_encoder.transform([UNK])[0])
                    else:
                        window_vec.append(words_encoder.transform([word])[0])
                feature_vecs.append([np.asarray(window_vec), tags_encoder.transform([tag])[0]])

            word_plus_1 = word_plus_2
            if it.has_next():
                word_plus_2 = it.next()[0]
            else:
                word_plus_2 = END

            word_minus_2, word_minus_1 = word_minus_1, tok
    return feature_vecs


# create a class encapsulating the network
class OurNetwork(object):
    # The init method adds parameters to the parameter collection.
    def __init__(self, pc, DIM, VOCAB_SIZE, out_dim, mid_dim):
        self.pW = pc.add_parameters((mid_dim, DIM * 5))
        self.pB_1 = pc.add_parameters(mid_dim)
        self.pV = pc.add_parameters((out_dim, mid_dim))
        self.pB_2 = pc.add_parameters(out_dim)
        self.E = pc.add_lookup_parameters((VOCAB_SIZE, DIM))

    # the __call__ method applies the network to an input
    def __call__(self, inputs):
        lookup = self.E
        emb_vectors = [lookup[i] for i in inputs]
        net_input = dy.concatenate(emb_vectors)
        net_output = dy.softmax(self.pV * (dy.tanh((self.pW * net_input) + self.pB_1)) + self.pB_2)
        return net_output

    def create_network_return_loss(self, inputs, expected_output):
        dy.renew_cg()
        out = self(inputs)
        loss = -dy.log(dy.pick(out, expected_output))
        return loss

    def create_network_return_best(self, inputs):
        dy.renew_cg()
        out = self(inputs)
        return dy.np.argmax(out.npvalue())


if __name__ == '__main__':
    corpus_file_location = 'pos/train'
    dev_file_location = 'pos/dev'

    # reading curpus_file
    train_sents = read_file(corpus_file_location)
    train_data = [pair for sent in train_sents for pair in sent]

    # adding the start-end special words
    train_data.append([START, START])
    train_data.append([END, END])
    train_data.append([UNK, UNK])

    # counting number of available tags and words
    tags_encoder = preprocessing.LabelEncoder()
    tags_encoder.fit([pair[1] for pair in train_data])
    words_encoder = preprocessing.LabelEncoder()
    words_encoder.fit([pair[0] for pair in train_data])

    # creating a dictionary to get the word index for each word
    embedded_location = defaultdict(int)

    # reset the global cg
    dy.renew_cg()

    # create parameter collection
    m = dy.ParameterCollection()

    # constants
    DIM = 50
    VOCAB_SIZE = len(words_encoder.classes_)
    out_dim = len(tags_encoder.classes_)
    mid_dim = int((DIM * 5 + out_dim) / 2)
    alpha = 0.1
    epochs = 15

    # create network
    network = OurNetwork(m, DIM=DIM, VOCAB_SIZE=VOCAB_SIZE, out_dim=out_dim, mid_dim=mid_dim)

    # create trainer
    trainer = dy.SimpleSGDTrainer(m)
    trainer.learning_rate = alpha

    dev_sents = read_file(dev_file_location)

    # train network
    for epoch in range(epochs):
        train_encoded = create_feature_vecs(random.choices(train_sents, k=200), words_encoder, tags_encoder)
        total_loss = 0
        seen_instances = 0
        for inp, lbl in train_encoded:
            loss = network.create_network_return_loss(inp, lbl)
            seen_instances += 1
            total_loss += loss.value()
            loss.backward()
            trainer.update()
            if (seen_instances > 1 and seen_instances % 500 == 0):
                print("average loss is:", total_loss / seen_instances)
        print("the learning rate is:", trainer.learning_rate)
        trainer.learning_rate = alpha * ((total_loss / seen_instances) ** 2)

        # checking accuracy on dev set
        encoded_dev_sents = create_feature_vecs(random.choices(dev_sents, k=20), words_encoder, tags_encoder)
        y_predicted = [x[0] for x in encoded_dev_sents]
        y_predicted = [network.create_network_return_best(y) for y in y_predicted]
        y_true = [x[1] for x in encoded_dev_sents]

        # NER Code
        # y_predicted = tags_encoder.inverse_transform(y_predicted)
        # y_true = tags_encoder.inverse_transform(y_true)
        #
        # counter = 0
        # currect_counter = 0
        #
        # for y, y_tag in zip(y_true, y_predicted):
        #     if y != 'O' and y_tag != 'O':
        #         if y == y_tag:
        #             currect_counter += 1
        #         counter += 1
        # try:
        #     acc = currect_counter / counter
        # except ZeroDivisionError:
        #     acc = 0.0

        # POS Code
        acc = accuracy_score(y_true, y_predicted, normalize=True, sample_weight=None)

        print("Accuracy:", acc)

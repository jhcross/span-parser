"""
Bi-LSTM network for span-based constituency parsing.
"""

from __future__ import print_function
from __future__ import division

import time
import random
import sys

import dynet
import numpy as np

from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser

class LSTM(object):
    """
    LSTM class with initial state as parameter, and all parameters
        initialized in [-0.01, 0.01].
    """

    number = 0

    def __init__(self, input_dims, output_dims, model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        self.W_i = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_i = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_f = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_f = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_c = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_c = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_o = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_o = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.c0 = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )

        self.W_params = [self.W_i, self.W_f, self.W_c, self.W_o]
        self.b_params = [self.b_i, self.b_f, self.b_c, self.b_o]
        self.params = self.W_params + self.b_params + [self.c0]
    
    class State(object):

        def __init__(self, lstm):
            self.lstm = lstm

            self.outputs = []

            self.c = dynet.parameter(self.lstm.c0)
            self.h = dynet.tanh(self.c)

            self.W_i = dynet.parameter(self.lstm.W_i)
            self.b_i = dynet.parameter(self.lstm.b_i)

            self.W_f = dynet.parameter(self.lstm.W_f)
            self.b_f = dynet.parameter(self.lstm.b_f)

            self.W_c = dynet.parameter(self.lstm.W_c)
            self.b_c = dynet.parameter(self.lstm.b_c)

            self.W_o = dynet.parameter(self.lstm.W_o)
            self.b_o = dynet.parameter(self.lstm.b_o)


        def add_input(self, input_vec):
            """
            Note that this function updates the existing State object!
            """
            x = dynet.concatenate([input_vec, self.h])

            i = dynet.logistic(self.W_i * x + self.b_i)
            f = dynet.logistic(self.W_f * x + self.b_f)
            g = dynet.tanh(self.W_c * x + self.b_c)
            o = dynet.logistic(self.W_o * x + self.b_o)

            c = dynet.cmult(f, self.c) + dynet.cmult(i, g)
            h = dynet.cmult(o, dynet.tanh(c))

            self.c = c
            self.h = h
            self.outputs.append(h)

            return self


        def output(self):
            return self.outputs[-1]


    def initial_state(self):
        return LSTM.State(self)




class Network(object):

    def __init__(
        self,
        word_count,
        tag_count,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        struct_out,
        label_out,
        droprate=0,
        struct_spans=4,
        label_spans=3,
    ):

        self.word_count = word_count
        self.tag_count = tag_count
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.struct_out = struct_out
        self.label_out = label_out

        self.droprate = droprate

        self.model = dynet.Model()

        self.trainer = dynet.AdadeltaTrainer(self.model, eps=1e-7, rho=0.99)
        random.seed(1)

        self.activation = dynet.rectify

        self.word_embed = self.model.add_lookup_parameters(
            (word_count, word_dims),
        )
        self.tag_embed = self.model.add_lookup_parameters(
            (tag_count, tag_dims),
        )

        self.fwd_lstm1 = LSTM(word_dims + tag_dims, lstm_units, self.model)
        self.back_lstm1 = LSTM(word_dims + tag_dims, lstm_units, self.model)

        self.fwd_lstm2 = LSTM(2 * lstm_units, lstm_units, self.model)
        self.back_lstm2 = LSTM(2 * lstm_units, lstm_units, self.model)


        self.struct_hidden_W = self.model.add_parameters(
            (hidden_units, 4 * struct_spans * lstm_units),
            dynet.UniformInitializer(0.01),
        )
        self.struct_hidden_b = self.model.add_parameters(
            (hidden_units,),
            dynet.ConstInitializer(0),
        )
        self.struct_output_W = self.model.add_parameters(
            (struct_out, hidden_units),
            dynet.ConstInitializer(0),
        )
        self.struct_output_b = self.model.add_parameters(
            (struct_out,),
            dynet.ConstInitializer(0),
        )

        self.label_hidden_W = self.model.add_parameters(
            (hidden_units, 4 * label_spans * lstm_units),
            dynet.UniformInitializer(0.01),
        )
        self.label_hidden_b = self.model.add_parameters(
            (hidden_units,),
            dynet.ConstInitializer(0),
        )
        self.label_output_W = self.model.add_parameters(
            (label_out, hidden_units),
            dynet.ConstInitializer(0),
        )
        self.label_output_b = self.model.add_parameters(
            (label_out,),
            dynet.ConstInitializer(0),
        )


    def init_params(self):

        self.word_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.word_embed.shape()),
        )
        self.tag_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.tag_embed.shape()),
        )


    def prep_params(self):

        self.W1_struct = dynet.parameter(self.struct_hidden_W)
        self.b1_struct = dynet.parameter(self.struct_hidden_b)

        self.W2_struct = dynet.parameter(self.struct_output_W)
        self.b2_struct = dynet.parameter(self.struct_output_b)        

        self.W1_label = dynet.parameter(self.label_hidden_W)
        self.b1_label = dynet.parameter(self.label_hidden_b)

        self.W2_label = dynet.parameter(self.label_output_W)
        self.b2_label = dynet.parameter(self.label_output_b)    


    def evaluate_recurrent(self, word_inds, tag_inds, test=False):

        fwd1 = self.fwd_lstm1.initial_state()
        back1 = self.back_lstm1.initial_state()

        fwd2 = self.fwd_lstm2.initial_state()
        back2 = self.back_lstm2.initial_state()

        sentence = []

        for (w, t) in zip(word_inds, tag_inds):
            wordvec = dynet.lookup(self.word_embed, w)
            tagvec = dynet.lookup(self.tag_embed, t)
            vec = dynet.concatenate([wordvec, tagvec])
            sentence.append(vec)

        fwd1_out = []
        for vec in sentence:
            fwd1 = fwd1.add_input(vec)
            fwd_vec = fwd1.output()
            fwd1_out.append(fwd_vec)

        back1_out = []
        for vec in reversed(sentence):
            back1 = back1.add_input(vec)
            back_vec = back1.output()
            back1_out.append(back_vec)

        lstm2_input = []
        for (f, b) in zip(fwd1_out, reversed(back1_out)):
            lstm2_input.append(dynet.concatenate([f, b]))

        fwd2_out = []
        for vec in lstm2_input:
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec, self.droprate)
            fwd2 = fwd2.add_input(vec)
            fwd_vec = fwd2.output()
            fwd2_out.append(fwd_vec)

        back2_out = []
        for vec in reversed(lstm2_input):
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec, self.droprate)
            back2 = back2.add_input(vec)
            back_vec = back2.output()
            back2_out.append(back_vec)

        fwd_out = [dynet.concatenate([f1, f2]) for (f1, f2) in zip(fwd1_out, fwd2_out)]
        back_out = [dynet.concatenate([b1, b2]) for (b1, b2) in zip(back1_out, back2_out)]

        return fwd_out, back_out[::-1]


    def evaluate_struct(self, fwd_out, back_out, lefts, rights, test=False):

        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
        fwd_span_vec = dynet.concatenate(fwd_span_out)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = dynet.concatenate(back_span_out)

        hidden_input = dynet.concatenate([fwd_span_vec, back_span_vec])

        if self.droprate > 0 and not test:
            hidden_input = dynet.dropout(hidden_input, self.droprate)

        hidden_output = self.activation(self.W1_struct * hidden_input + self.b1_struct)

        scores = (self.W2_struct * hidden_output + self.b2_struct)

        return scores



    def evaluate_label(self, fwd_out, back_out, lefts, rights, test=False):

        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index - 1])
        fwd_span_vec = dynet.concatenate(fwd_span_out)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = dynet.concatenate(back_span_out)

        hidden_input = dynet.concatenate([fwd_span_vec, back_span_vec])

        if self.droprate > 0 and not test:
            hidden_input = dynet.dropout(hidden_input, self.droprate)

        hidden_output = self.activation(self.W1_label * hidden_input + self.b1_label)

        scores = (self.W2_label * hidden_output + self.b2_label)

        return scores


    def save(self, filename):
        """
        Appends architecture hyperparameters to end of dynet model file.
        """
        self.model.save(filename)

        with open(filename, 'a') as f:
            f.write('\n')
            f.write('word_count = {}\n'.format(self.word_count))
            f.write('tag_count = {}\n'.format(self.tag_count))
            f.write('word_dims = {}\n'.format(self.word_dims))
            f.write('tag_dims = {}\n'.format(self.tag_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('struct_out = {}\n'.format(self.struct_out))
            f.write('label_out = {}\n'.format(self.label_out))


    @staticmethod
    def load(filename):
        """
        Loads file created by save() method.
        """
        with open(filename) as f:
            f.readline()
            f.readline()
            word_count = int(f.readline().split()[-1])
            tag_count = int(f.readline().split()[-1])
            word_dims = int(f.readline().split()[-1])
            tag_dims = int(f.readline().split()[-1])
            lstm_units = int(f.readline().split()[-1])
            hidden_units = int(f.readline().split()[-1])
            struct_out = int(f.readline().split()[-1])
            label_out = int(f.readline().split()[-1])

        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=struct_out,
            label_out=label_out,
        )
        network.model.load(filename)

        return network


    @staticmethod
    def train(
        feature_mapper,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        epochs,
        batch_size,
        train_data_file,
        dev_data_file,
        model_save_file,
        droprate,
        unk_param,
        alpha=1.0,
        beta=0.0,
    ):

        start_time = time.time()

        fm = feature_mapper
        word_count = fm.total_words()
        tag_count = fm.total_tags()

        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=2,
            label_out=fm.total_label_actions(),
            droprate=droprate,
        )
        network.init_params()

        print('Hidden units: {},  per-LSTM units: {}'.format(
            hidden_units,
            lstm_units,
        ))
        print('Embeddings: word={}  tag={}'.format(
            (word_count, word_dims),
            (tag_count, tag_dims),
        ))
        print('Dropout rate: {}'.format(droprate))
        print('Parameters initialized in [-0.01, 0.01]')
        print('Random UNKing parameter z = {}'.format(unk_param))
        print('Exploration: alpha={} beta={}'.format(alpha, beta))

        training_data = fm.gold_data_from_file(train_data_file)
        num_batches = -(-len(training_data) // batch_size) 
        print('Loaded {} training sentences ({} batches of size {})!'.format(
            len(training_data),
            num_batches,
            batch_size,
        ))
        parse_every = -(-num_batches // 4)

        dev_trees = PhraseTree.load_treefile(dev_data_file)
        print('Loaded {} validation trees!'.format(len(dev_trees)))

        best_acc = FScore()

        for epoch in xrange(1, epochs + 1):
            print('........... epoch {} ...........'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            np.random.shuffle(training_data)

            for b in xrange(num_batches):
                batch = training_data[(b * batch_size) : ((b + 1) * batch_size)]

                explore = [
                    Parser.exploration(
                        example,
                        fm,
                        network,
                        alpha=alpha,
                        beta=beta,
                    ) for example in batch
                ]
                for (_, acc) in explore:
                    training_acc += acc

                batch = [example for (example, _) in explore]

                dynet.renew_cg()
                network.prep_params()

                errors = []

                for example in batch:

                    ## random UNKing ##
                    for (i, w) in enumerate(example['w']):
                        if w <= 2:
                            continue

                        freq = fm.word_freq_list[w]
                        drop_prob = unk_param / (unk_param + freq)
                        r = np.random.random()
                        if r < drop_prob:
                            example['w'][i] = 0

                    fwd, back = network.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )

                    for (left, right), correct in example['struct_data'].items():
                        scores = network.evaluate_struct(fwd, back, left, right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs, correct))
                        errors.append(loss)
                    total_states += len(example['struct_data'])

                    for (left, right), correct in example['label_data'].items():
                        scores = network.evaluate_label(fwd, back, left, right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs, correct))
                        errors.append(loss)
                    total_states += len(example['label_data'])

                batch_error = dynet.esum(errors)
                total_cost += batch_error.scalar_value()
                batch_error.backward()
                network.trainer.update()

                mean_cost = total_cost / total_states

                print(
                    '\rBatch {}  Mean Cost {:.4f} [Train: {}]'.format(
                        b,
                        mean_cost,
                        training_acc,
                    ),
                    end='',
                )
                sys.stdout.flush()

                if ((b + 1) % parse_every) == 0 or b == (num_batches - 1):
                    dev_acc = Parser.evaluate_corpus(
                        dev_trees,
                        fm,
                        network,
                    )
                    print('  [Val: {}]'.format(dev_acc))

                    if dev_acc > best_acc:
                        best_acc = dev_acc 
                        network.save(model_save_file)
                        print('    [saved model: {}]'.format(model_save_file)) 

            current_time = time.time()
            runmins = (current_time - start_time)/60.
            print('  Elapsed time: {:.2f}m'.format(runmins))


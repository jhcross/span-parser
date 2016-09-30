
from __future__ import print_function
from __future__ import division

import sys
import json
from collections import defaultdict, OrderedDict

import numpy as np

from phrase_tree import PhraseTree
from parser import Parser




class FeatureMapper(object):
    """
    Maps words, tags, and label actions to indices.
    """

    UNK = '<UNK>'
    START = '<s>'
    STOP = '</s>'


    @staticmethod
    def vocab_init(fname, verbose=True):
        """
        Learn vocabulary from file of strings.
        """
        word_freq = defaultdict(int)
        tag_freq = defaultdict(int)
        label_freq = defaultdict(int)

        trees = PhraseTree.load_treefile(fname)

        for i, tree in enumerate(trees):
            for (word, tag) in tree.sentence:
                word_freq[word] += 1
                tag_freq[tag] += 1

            for action in Parser.gold_actions(tree):
                if action.startswith('label-'):
                    label = action[6:]
                    label_freq[label] += 1 

            if verbose:
                print('\rTree {}'.format(i), end='')
                sys.stdout.flush()

        if verbose:
            print('\r', end='')


        words = [
            FeatureMapper.UNK,
            FeatureMapper.START,
            FeatureMapper.STOP,
        ] + sorted(word_freq)         
        wdict = OrderedDict((w,i) for (i,w) in enumerate(words))

        tags = [
            FeatureMapper.UNK,
            FeatureMapper.START,
            FeatureMapper.STOP,
        ] +  sorted(tag_freq)
        tdict = OrderedDict((t,i) for (i,t) in enumerate(tags))

        labels = sorted(label_freq)
        ldict = OrderedDict((l,i) for (i,l) in enumerate(labels))

        if verbose:
            print('Loading features from {}'.format(fname))
            print('({} words, {} tags, {} nonterminal-chains)'.format(
                len(wdict),
                len(tdict),
                len(ldict),
            ))
        
        return {
            'wdict': wdict,
            'word_freq': word_freq,
            'tdict': tdict,
            'ldict': ldict,
        }


    def __init__(self, vocabfile, verbose=True):

        if vocabfile is not None:
            data = FeatureMapper.vocab_init(
                fname=vocabfile,
                verbose=verbose,
            )
            self.wdict = data['wdict']
            self.word_freq = data['word_freq']
            self.tdict = data['tdict']
            self.ldict = data['ldict']

            self.word_freq_list = []
            for word in self.wdict.keys():
                if word in self.word_freq:
                    self.word_freq_list.append(self.word_freq[word])
                else:
                    self.word_freq_list.append(0)


    @staticmethod
    def from_dict(data):
        new = FeatureMapper(None)
        new.wdict = data['wdict']
        new.word_freq = data['word_freq']
        new.tdict = data['tdict']
        new.ldict = data['ldict']
        new.word_freq_list = data['word_freq_list']
        return new


    def as_dict(self):
        return {
            'wdict': self.wdict,
            'word_freq': self.word_freq,
            'tdict': self.tdict,
            'ldict': self.ldict,
            'word_freq_list': self.word_freq_list
        }


    def save_json(self, filename):
        with open(filename, 'w') as fh:
            json.dump(self.as_dict(), fh)


    @staticmethod
    def load_json(filename):
        with open(filename) as fh:
            data = json.load(fh, object_pairs_hook=OrderedDict)
        return FeatureMapper.from_dict(data)


    def total_words(self):
        return len(self.wdict)


    def total_tags(self):
        return len(self.tdict)


    def total_label_actions(self):
        return 1 + len(self.ldict)


    def s_action_index(self, action):
        if action == 'sh':
            return 0
        elif action == 'comb':
            return 1
        else:
            raise ValueError('Not s-action: {}'.format(action))


    def l_action_index(self, action):
        if action == 'none':
            return 0
        elif action.startswith('label-'):
            label = action[6:]
            label_index = self.ldict.get(label, None)
            if label_index is not None:
                return 1 + label_index
            else:
                return 0
        else:
            raise ValueError('Not l-action: {}'.format(action))


    def s_action(self, index):
        return ('sh', 'comb')[index]


    def l_action(self, index):
        if index == 0:
            return 'none'
        else:
            return 'label-' + self.ldict.keys()[index - 1]


    def sentence_sequences(self, sentence):
        """
        Array of indices for words and tags.
        """
        sentence = (
            [(FeatureMapper.START, FeatureMapper.START)] + 
            sentence + 
            [(FeatureMapper.STOP, FeatureMapper.STOP)]
        )

        words = [
            self.wdict[w] 
            if w in self.wdict else self.wdict[FeatureMapper.UNK]
            for (w, t) in sentence
        ]
        tags = [
            self.tdict[t] 
            if t in self.tdict else self.tdict[FeatureMapper.UNK]
            for (w, t) in sentence
        ]

        w = np.array(words).astype('int32')
        t = np.array(tags).astype('int32')

        return w, t


    def gold_data(self, reftree):
        """
        Static oracle for tree.
        """

        w, t = self.sentence_sequences(reftree.sentence)

        (s_features, l_features) = Parser.training_data(reftree)

        struct_data = {}
        for (features, action) in s_features:
            struct_data[features] = self.s_action_index(action)

        label_data = {}
        for (features, action) in l_features:
            label_data[features] = self.l_action_index(action)

        return {
            'tree': reftree,
            'w': w,
            't': t,
            'struct_data': struct_data,
            'label_data': label_data,
        }


    def gold_data_from_file(self, fname):
        """
        Static oracle for file.
        """
        trees = PhraseTree.load_treefile(fname)
        result = []
        for tree in trees:
            sentence_data = self.gold_data(tree)
            result.append(sentence_data)
        return result


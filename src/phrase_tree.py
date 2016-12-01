#!/usr/bin/env python

"""
Recursive representation of a phrase-structure parse tree 
    for natural language sentences.
"""

from __future__ import print_function
from __future__ import division

from collections import defaultdict

class PhraseTree(object):

    puncs = [",", ".", ":", "``", "''", "PU"] ## (COLLINS.prm)


    def __init__(
        self, 
        symbol=None, 
        children=[],
        sentence=[], 
        leaf=None,
    ):
        self.symbol = symbol        # label at top node
        self.children = children    # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf            # word at bottom level else None
        
        self._str = None


    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1], 
                    self.sentence[self.leaf][0],
                )
        return self._str


    def propagate_sentence(self, sentence):
        """
        Recursively assigns sentence (list of (word, POS) pairs)
            to all nodes of a tree.
        """
        self.sentence = sentence
        for child in self.children:
            child.propagate_sentence(sentence)

        
    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                    self.symbol, 
                    self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result


    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        if t.symbol == 'TOP' and len(t.children) == 1:
            t = t.children[0]
        
        return t


    @staticmethod
    def _parse(line, index, sentence):
        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."
                
        assert line[index] == '(', "Invalid tree string {} at {}".format(line, index)
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index, sentence)
                children.append(t)
                
            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)
                
                    symbol = line[index:rpos] # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1
                    index = rpos
                
            if line[index] == " ":
                index += 1

        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        t = PhraseTree(
            symbol=symbol, 
            children=children, 
            sentence=sentence,
            leaf=leaf,
        )

        return (index + 1), t


    def left_span(self):
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is not None:
                self._left_span = self.leaf
            else:
                self._left_span = self.children[0].left_span()
            return self._left_span


    def right_span(self):
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is not None:
                self._right_span = self.leaf
            else:
                self._right_span = self.children[-1].right_span()
            return self._right_span


    def brackets(self, advp_prt=True, counts=None):
        
        if counts is None:
            counts = defaultdict(int)

        if self.leaf is not None:
            return {}

        nonterm = self.symbol
        if advp_prt and nonterm=='PRT':
            nonterm = 'ADVP'

        left = self.left_span()
        right = self.right_span()

        # ignore punctuation
        while (
            left < len(self.sentence) and 
            self.sentence[left][1] in PhraseTree.puncs
        ):
            left += 1
        while (
            right > 0 and self.sentence[right][1] in PhraseTree.puncs
        ):
            right -= 1

        if left <= right and nonterm != 'TOP':
            counts[(nonterm, left, right)] += 1
        
        for child in self.children:
            child.brackets(advp_prt=advp_prt, counts=counts)

        return counts


    def phrase(self):
        if self.leaf is not None:
            return [(self.leaf, self.symbol)]
        else:
            result = []
            for child in self.children:
                result.extend(child.phrase())
            return result
        

    @staticmethod
    def load_treefile(fname):
        trees = []
        for line in open(fname):
            t = PhraseTree.parse(line)
            trees.append(t)
        return trees


    def compare(self, gold, advp_prt=True):
        """
        returns (Precision, Recall, F-measure)
        """
        predbracks = self.brackets(advp_prt)
        goldbracks = gold.brackets(advp_prt)

        correct = 0
        for gb in goldbracks:
            if gb in predbracks:
                correct += min(goldbracks[gb], predbracks[gb])

        pred_total = sum(predbracks.values())
        gold_total = sum(goldbracks.values())

        return FScore(correct, pred_total, gold_total)


    def enclosing(self, i, j):
        """
        Returns the left and right indices of the labeled span in the tree
            which is next-larger than (i, j)
            (whether or not (i, j) is itself a labeled span)
        """
        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                if (left == i) and (right == j):
                    break
                return child.enclosing(i, j)

        return (self.left_span(), self.right_span())


    def span_labels(self, i, j):
        """
        Returns a list of span labels (if any) for (i, j)
        """
        if self.leaf is not None:
            return []
            
        if (self.left_span() == i) and (self.right_span() == j):
            result = [self.symbol]
        else:
            result = []

        for child in self.children:
            left = child.left_span()
            right = child.right_span()
            if (left <= i) and (right >= j):
                result.extend(child.span_labels(i, j))
                break

        return result




class FScore(object):

    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct        # correct brackets
        self.predcount = predcount    # total predicted brackets
        self.goldcount = goldcount    # total gold brackets


    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0


    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0


    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0
   

    def __str__(self):
        precision = self.precision()
        recall    = self.recall()
        fscore = self.fscore()
        return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision, 
            recall, 
            fscore,
        )


    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self


    def __add__(self, other):
        return Fmeasure(self.correct + other.correct, 
                        self.predcount + other.predcount, 
                        self.goldcount + other.goldcount)


    def __cmp__(self, other):
        return cmp(self.fscore(), other.fscore())


    @staticmethod
    def parseval(gold_file, test_file):
        gold_trees = PhraseTree.load_treefile(gold_file)
        test_trees = PhraseTree.load_treefile(test_file)
        cumulative = FScore()

        for gold, test in zip(gold_trees, test_trees):
            acc = test.compare(gold, advp_prt=True)
            cumulative += acc

        return cumulative





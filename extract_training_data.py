# from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
from tqdm import tqdm
from typing import Tuple, List

import keras
import numpy as np

from dep_utils import conll_reader


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add((self.buffer[-1], self.stack.pop(), label))

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add((parent, self.buffer.pop(), label))
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)


def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None

    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):
    deprels = dep_structure.deprels

    sorted_nodes = [k for k, v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident, node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id] -= 1
            seq.append((copy.deepcopy(state), ("left_arc", stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id] -= 1
            seq.append((copy.deepcopy(state), ("right_arc", bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
    return seq


# dep_relations_old = dep_relations = [
#     "tmod", "vmod","csubjpass","rcmod","ccomp","poss","parataxis","appos","dep","iobj","pobj","mwe","quantmod","acomp","number","csubj","root","auxpass","prep","mark","expl","cc","npadvmod","prt","nsubj","advmod","conj","advcl","punct","aux","pcomp","discourse","nsubjpass","predet","cop","possessive","nn","xcomp","preconj","num","amod","dobj","neg","dt","det"]


def read_dep_relations():
    dep_relations = []

    input_files = ['data/train.conll', 'data/dev.conll', 'data/test.conll']
    for input_file in input_files:
        with open(input_file, 'r') as f:
            train_trees = list(conll_reader(f))
        for tree in train_trees:
            for k, v in tree.deprels.items():
                if v.deprel not in dep_relations:
                    dep_relations.append(v.deprel)

    return dep_relations

dep_relations = read_dep_relations()



class FeatureExtractor(object):
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()
        self.output_format = 'pt' # 'pt' or 'tf'

    def make_output_labels(self):
        labels = []
        labels.append(("shift", None))

        for rel in dep_relations:
            labels.append(("left_arc", rel))
            labels.append(("right_arc", rel))
        return dict((label, index) for (index, label) in enumerate(labels))

    def read_vocab(self, vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_input_representation(self, words, pos, state):
        # self.word_vocab is a dictionary of words and their indices
        # state contains two lists: stack and buffer
        # return a numpy array of the indices of the first three words in the stack and buffer
        # if there are less than three words in the stack or buffer, pad with the index for "<NULL>"
        rep = np.zeros(6)
        for i in range(3):
            if len(state.stack) > i:
                idx = state.stack[-i - 1]
                word = words[idx] if not words[idx] else words[idx].lower()
                if word in self.word_vocab:
                    rep[i] = self.word_vocab[word]
                elif word is None:
                    rep[i] = self.word_vocab["<ROOT>"]
                elif pos[idx] in self.word_vocab:
                    rep[i] = self.word_vocab[pos[idx]]
                else:
                    rep[i] = self.word_vocab["<UNK>"]
            else:
                rep[i] = self.word_vocab["<NULL>"]
            if len(state.buffer) > i:
                idx = state.buffer[-i - 1]
                word = words[idx] if not words[idx] else words[idx].lower()
                if word in self.word_vocab:
                    rep[i + 3] = self.word_vocab[word]
                elif word is None:
                    rep[i + 3] = self.word_vocab["<ROOT>"]
                elif pos[idx] in self.word_vocab:
                    rep[i + 3] = self.word_vocab[pos[idx]]
                else:
                    rep[i + 3] = self.word_vocab["<UNK>"]
            else:
                rep[i + 3] = self.word_vocab["<NULL>"]
        return rep

    def get_output_representation(self, output_pair):
        # each output_pair is a tuple of (transition, label)
        # there are three possible transitions: "shift", "left_arc", "right_arc"
        # there are 45 possible labels, all included in the dep_relations list
        # return a numpy array of length 91
        if self.output_format == 'pt':
            return np.array(self.output_labels[output_pair])
        elif self.output_format == 'tf':
            return keras.utils.to_categorical(self.output_labels[output_pair], len(self.output_labels)) # xy


def get_training_matrices(extractor, input_filename: str, n=np.inf) -> Tuple[List, List]:
    inputs = []
    outputs = []
    count = 0
    with open(input_filename, "r") as in_file:
        dtrees = list(conll_reader(in_file))
    for dtree in tqdm(dtrees, total=min(len(dtrees), n)):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        count += 1
        if count >= n:
            break
    return inputs, outputs


if __name__ == "__main__":
    # print("Loading data... (this might take a minute)")
    WORD_VOCAB_FILE = "data/words.vocab"
    POS_VOCAB_FILE = "data/pos.vocab"

    try:
        word_vocab_f = open(WORD_VOCAB_FILE, "r")
        pos_vocab_f = open(POS_VOCAB_FILE, "r")
    except FileNotFoundError:
        print(
            "Could not find vocabulary files {} and {}".format(
                WORD_VOCAB_FILE, POS_VOCAB_FILE
            )
        )
        sys.exit(1)

    input_file = 'data/train.conll'
    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Starting feature extraction...")

    # prepare two versions of target
    extractor.output_format = 'pt'
    inputs, outputs_pt = get_training_matrices(extractor, in_file)
    inputs = np.stack(inputs)
    outputs_pt = np.stack(outputs_pt)
    np.save("data/target_train.npy", outputs_pt)

    extractor.output_format = 'tf'
    _, outputs_tf = get_training_matrices(extractor, input_file)
    outputs_tf = np.stack(outputs_tf)
    np.save("data/target_train_tf.npy", outputs_tf)

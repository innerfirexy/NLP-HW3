from dep_utils import conll_reader, DependencyTree
import copy
from pprint import pprint
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        ### START YOUR CODE ###
        self.stack.append(self.buffer.pop())
        ### END YOUR CODE ###

    def left_arc(self, label: str):
        assert len(self.stack) >= 2
        ### START YOUR CODE ###
        top1 = self.stack.pop()
        top2 = self.stack.pop()
        self.deps.add((top1, top2, label))
        self.stack.append(top1)
        ### END YOUR CODE ###

    def right_arc(self, label: str):
        assert len(self.stack) >= 2
        ### START YOUR CODE ###
        top1 = self.stack.pop()
        top2 = self.stack.pop()
        self.deps.add((top2, top1, label))
        self.stack.append(top2)
        ### END YOUR CODE ###

    def __repr__(self):
        return "({},{},{})".format(self.stack, self.buffer, self.deps)


class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_tree: DependencyTree) -> List[Tuple[State, Tuple[str, str]]]:
    deprels = dep_tree.deprels

    word_ids = list(deprels.keys())
    state = State(word_ids)
    state.stack.append(0) # ROOT

    childcount = defaultdict(int)
    for _, rel in deprels.items():
        childcount[rel.head] += 1

    seq = []
    while len(state.buffer) > 0 or len(state.stack) > 1:
        if state.stack[-1] == 0:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue
        
        stack_top1 = deprels[state.stack[-1]]
        if state.stack[-2] == 0:
            stack_top2 = RootDummy()
        else:
            stack_top2 = deprels[state.stack[-2]]

        # Decide transition action
        ### START YOUR CODE ###
        if stack_top1.id == stack_top2.head: # Left-Arc
            childcount[stack_top1.id] -= 1
            label = stack_top2.deprel
            seq.append((copy.deepcopy(state), ("left_arc", label)))
            state.left_arc(label)

        elif stack_top1.head == stack_top2.id and childcount[stack_top1.id] == 0: # Right-Arc
            childcount[stack_top2.id] -= 1
            label = stack_top1.deprel
            seq.append((copy.deepcopy(state), ("right_arc", label)))
            state.right_arc(label)

        else: # Shift
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
        ### END YOUR CODE ###
    
    seq.append((copy.deepcopy(state), ("done", None)))

    return seq
from dep_utils import DependencyArc, DependencyTree, conll_reader
import sys

import numpy as np
import tensorflow as tf
import keras
import torch
from model import BaseModel as PyTorchModel

from extract_training_data import FeatureExtractor, State

tf.compat.v1.disable_eager_execution()


class Parser(object):
    def __init__(self, extractor, modelfile, backend='pt'):
        if backend == 'pt':
            self.pt_model = PyTorchModel(
                word_vocab_size=len(extractor.word_vocab),
                output_size=len(extractor.output_labels)
            )
            self.pt_model.load_state_dict(torch.load(modelfile))
        elif backend == 'tf':
            self.model = keras.models.load_model(modelfile)
        self.backend = backend
        
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.output_labels.items()]
        )

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            current_state = self.extractor.get_input_representation(words, pos, state)

            if self.backend == 'pt':
                with torch.no_grad():
                    input_tensor = torch.tensor(current_state.reshape(1, 6), dtype=torch.long)
                    prediction = self.pt_model(input_tensor).detach().numpy()[0]
            elif self.backend == 'tf':
                prediction = self.model.predict(current_state.reshape(1, 6))[0]

            sorted_indices = np.argsort(prediction)[::-1]
            best_action = None
            for action_index in sorted_indices:
                best_action = self.output_labels[action_index]
                if best_action[0] == "shift" and (
                    len(state.buffer) > 1 or len(state.stack) == 0
                ):
                    break
                elif (
                    best_action[0] == "left_arc"
                    and len(state.stack) > 0
                    and state.stack[-1] != 0
                ):
                    break
                elif best_action[0] == "right_arc" and len(state.stack) > 0:
                    break

            # apply the best action to the state
            if best_action[0] == "shift":
                state.shift()
            elif best_action[0] == "left_arc":
                state.left_arc(best_action[1])
            elif best_action[0] == "right_arc":
                state.right_arc(best_action[1])

        tree = DependencyTree()
        for head_id, dep_id, rel in state.deps:
            tree.add_deprel(DependencyArc(dep_id, words[dep_id], pos[dep_id], head_id, rel))

        return tree


if __name__ == "__main__":
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

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, "data/model.h5")

    with open("data/dev.conll", "r") as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

    # extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    # parser = Parser(extractor, sys.argv[1])

    # with open(sys.argv[2], "r") as in_file:
    #     for dtree in conll_reader(in_file):
    #         words = dtree.words()
    #         pos = dtree.pos()
    #         deps = parser.parse_sentence(words, pos)
    #         print(deps.print_conll())
    #         print()

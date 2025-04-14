from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import tensorflow as tf
import keras
import torch
from train_pt import Model as PyTorchModel

from extract_training_data import FeatureExtractor, State

tf.compat.v1.disable_eager_execution()


class Parser(object):
    def __init__(self, extractor, modelfile, backend='pt'):
        if backend == 'pt':
            self.pt_model = PyTorchModel(
                word_vocab_size=len(extractor.word_vocab),
                pos_vocab_size=len(extractor.pos_vocab),
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
            # TODO: Write the body of this loop for part 4
            # first, use the feature extractor to obtain a representation of the current state
            # then, call model.predict(features) and retrieve a softmax actived vector of possible actions
            # Unfortunately, it is possible that the highest scoring transition is not possible. arc-left or arc-right are not permitted the stack is empty. Shifting the only word out of the buffer is also illegal, unless the stack is empty. Finally, the root node must never be the target of a left-arc. We should find the highest scoring transition that is legal, and apply it to the state.
            # print("state", state)
            # print("words", words)
            # print("pos", pos)
            # print("self.extractor.get_input_representation(state, words, pos)", self.extractor.get_input_representation(state, words, pos))
            # print("self.model.predict(self.extractor.get_input_representation(state, words, pos))", self.model.predict(self.extractor.get_input_representation(state, words, pos)))
            current_state = self.extractor.get_input_representation(words, pos, state)
            # print("current_state.shape", current_state.shape)
            # print("current_state", current_state)

            if self.backend == 'pt':
                with torch.no_grad():
                    input_tensor = torch.tensor(current_state.reshape(1, 6), dtype=torch.long)
                    prediction = self.pt_model(input_tensor).detach().numpy()[0]
            elif self.backend == 'tf':
                prediction = self.model.predict(current_state.reshape(1, 6))[0]


            # print("prediction.shape", prediction.shape)
            # print("prediction", prediction)
            # print("=====================================")
            # iteratively find the highest scoring transition that is legal

            # best_action_index = np.argmax(prediction)
            # best_action = self.output_labels[best_action_index]
            # while True:
            #     if best_action[0] == "shift" and (
            #         len(state.buffer) >= 1 or len(state.stack) == 0
            #     ):
            #         break
            #     elif (
            #         best_action[0] == "left_arc"
            #         and len(state.stack) > 0
            #         and state.stack[-1] != 0
            #     ):
            #         break
            #     elif best_action[0] == "right_arc" and len(state.stack) > 0:
            #         break
            #     else:
            #         prediction[best_action_index] = 0
            #         best_action_index = np.argmax(prediction)
            #         best_action = self.output_labels[best_action_index]

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

        result = DependencyStructure()
        for p, c, r in state.deps:
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result


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

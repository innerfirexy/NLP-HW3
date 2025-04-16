from extract_training_data import FeatureExtractor
import sys
import numpy as np
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import BaseModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', default='data/input_train.npy')
argparser.add_argument('--target_file', default='data/target_train.npy')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')


if __name__ == "__main__":
    args = argparser.parse_args()
    # WORD_VOCAB_FILE = "data/words.vocab"
    # POS_VOCAB_FILE = "data/pos.vocab"

    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab} and {args.pos_vocab}')
        sys.exit(1)
    
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file)
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.output_labels)

    model = BaseModel(word_vocab_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    inputs = np.load(args.input_file)
    targets = np.load(args.target_file) # pytorch input is int
    print("Done loading data.")

    # Train loop
    n_epochs = 5
    batch_size = 100
    print_loss_every = 100 # every 100 batches

    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(inputs) // batch_size

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            batch_count += 1
            if batch_count % print_loss_every == 0:
                avg_loss = epoch_loss / batch_count 
                sys.stdout.write(f'\rEpoch {epoch+1}/{n_epochs} - Batch {batch_count}/{n_batches} - Loss: {avg_loss:.4f}')
                sys.stdout.flush()
        # print
        print()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}')
    
    # save model
    now = datetime.datetime.now()
    torch.save(model.state_dict(), f'model_{now}.pt')
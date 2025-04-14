from extract_training_data import FeatureExtractor
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Model(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(word_vocab_size, 32)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * 6, 100)  # 32 * input_length(6)
        self.dense2 = nn.Linear(100, 10)
        self.dense3 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()
        self.output = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.output(self.dense3(x))
        return x



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
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.output_labels)

    model = Model(word_vocab_size, pos_vocab_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    inputs = np.load("data/input_train.npy")
    targets = np.load("data/target_train_int.npy") # pytorch input is int
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
    torch.save(model.state_dict(), "model.pt")
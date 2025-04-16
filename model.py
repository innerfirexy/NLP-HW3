import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
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
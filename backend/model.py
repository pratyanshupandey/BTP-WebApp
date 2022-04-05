from torch import nn

class Model(nn.Module):
    def __init__(self, word_vocab_size=1000, slot_vocab_size=127, intent_vocab_size=18):
        super(Model, self).__init__()
        self.word_embeddings = nn.Embedding(word_vocab_size, 128)
        self.lstm = nn.LSTM(128, 64)
        self.fc = nn.Linear(64, intent_vocab_size)
    
    def forward(self, input_sequence):
        embedded = self.word_embeddings(input_sequence)
        lstm_output, _ = self.lstm(embedded.view(len(input_sequence), 1, -1))
        fc_out = self.fc(lstm_output[-1])

        return fc_out
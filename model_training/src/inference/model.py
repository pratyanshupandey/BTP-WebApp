import torch
import torch.nn as nn

class TorchGRUIntent(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size) -> None:
        super(TorchGRUIntent, self).__init__()
        
        self.GRU = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)

    
    def forward(self, x):
        # x has shape L, N , H_in(emb)
        out, h = self.GRU(x)
        
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        # h has shape D∗num_layers,N,Hout(hidden size)​
        outputs = self.classifier(torch.mean(out, dim=0))

        # outputs has shape L * N * vocab
        return outputs


class TorchGRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size) -> None:
        super(TorchGRU, self).__init__()
        
        self.GRU = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)

    
    def forward(self, x):
        # x has shape L, N , H_in(emb)
        out, h = self.GRU(x)
        
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        outputs = self.classifier(out)

        # outputs has shape L * N * vocab
        return outputs

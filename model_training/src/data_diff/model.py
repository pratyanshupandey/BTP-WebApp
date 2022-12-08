import torch
import torch.nn as nn

class TorchGRUDiff(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size) -> None:
        super(TorchGRUDiff, self).__init__()
        
        self.GRU = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)

    
    def forward(self, x):
        # x has shape L, N , H_in(emb)
        out, h = self.GRU(x)
        
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        # h has shape D∗num_layers,N,Hout(hidden size)​
        outputs = self.classifier(torch.mean(out, dim=0))

        # outputs has shape N * vocab
        return outputs


class TorchLinearDiff(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size) -> None:
        super(TorchLinearDiff, self).__init__()
        
        self.model = nn.Sequential(
                        nn.Linear(in_features=embedding_size, out_features=hidden_size),
                        nn.Dropout(p=0.1),
                        nn.ReLU(),
                        nn.Linear(in_features=hidden_size, out_features=vocab_size)
                        )
    
    def forward(self, x):
        # x has shape L, N , H_in(emb)
        outputs = self.model(torch.mean(x, dim = 0))
        return outputs

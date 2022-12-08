import torch
import torch.nn as nn
from transformers import BertModel

class TorchGRU(nn.Module):
    def __init__(self, hidden_size, vocab_size) -> None:
        super(TorchGRU, self).__init__()
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.GRU = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)

    
    def forward(self, ids, mask):

        with torch.no_grad():
            x = self.bert_model(input_ids=ids, attention_mask=mask).last_hidden_state
            # x has shape N , L , 768

            x = torch.transpose(x, 0 , 1)
            # x has shape L, N , 768

        # x has shape L, N , H_in(emb)
        out, h = self.GRU(x)
        
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        outputs = self.classifier(out)

        # outputs has shape L * N * vocab
        return outputs


class TorchLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size) -> None:
        super(TorchLSTM, self).__init__()
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.LSTM = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=2, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(in_features=2*hidden_size, out_features=vocab_size)

    
    def forward(self, ids, mask):

        with torch.no_grad():
            x = self.bert_model(input_ids=ids, attention_mask=mask).last_hidden_state
            # x has shape N , L , 768

            x = torch.transpose(x, 0 , 1)
            # x has shape L, N , 768

        # x has shape L, N , H_in(emb)
        out, h = self.LSTM(x)
        
        # outhas size L(seq_len),N(batch),D∗H_out(hidden)
        outputs = self.classifier(out)

        # outputs has shape L * N * vocab
        return outputs


class Transformer(nn.Module):
    def __init__(self, vocab_size) -> None:
        super(Transformer, self).__init__()
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=2)
        self.classifier = nn.Linear(in_features=768, out_features=vocab_size)

    
    def forward(self, ids, mask, tgt):

        with torch.no_grad():
            x = self.bert_model(input_ids=ids, attention_mask=mask).last_hidden_state
            # x has shape N , L , 768

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(torch.device('cuda'))
        out = self.decoder(tgt=tgt, memory=x, tgt_mask=tgt_mask, memory_key_padding_mask=(mask == False))
        # out N * 1 * 768
        outputs = self.classifier(torch.squeeze(out))

        # outputs has shape N * vocab
        return outputs
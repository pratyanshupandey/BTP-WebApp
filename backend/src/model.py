import torch
import torch.nn as nn
from transformers import BertModel

class IntentModel(nn.Module):
    def __init__(self, labels_count) -> None:
        super(IntentModel, self).__init__()
        
        self.bert_model = BertModel(torch.load("models/bert_config.pt"))
        # torch.save(self.bert_model.config, "bert_config.pt")
        self.classifier = nn.Linear(in_features=768, out_features=labels_count)
    
    def forward(self, ids):
        out = self.bert_model(**ids).last_hidden_state
        outputs_tgt = self.classifier(torch.mean(out, dim=1))
        intent = torch.argmax(outputs_tgt, dim=-1)
        return intent

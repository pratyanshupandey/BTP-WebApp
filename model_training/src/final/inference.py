from torch import load
from model import IntentModel
from transformers import BertTokenizer
import json

class Inference:
    def __init__(self, path) -> None:
        file = open(path + "intents.json")
        self.intents = json.load(file)["intents"]

        self.model = IntentModel(len(self.intents))
        self.model.load_state_dict(load(path + "model.pt"))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    

    def calc_intent(self, text):
        ids = self.tokenizer.tokenize(text)
        intent_id = self.model(ids)
        return self.intents[intent_id]



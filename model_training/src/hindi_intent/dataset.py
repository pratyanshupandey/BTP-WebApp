import torch
from torch.utils.data import Dataset
import torchtext
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import json
from itertools import chain
import re
from transformers import AutoTokenizer, AutoModel


class BTP(Dataset):

    # BERT Tokenizer
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')

    # pad id
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    def __init__(self, path, mode = "intent", mlb = None):

        self.mode = mode

        with open(path, 'r') as f:
            data = json.loads(f.read())

            self.sentences = [e['dialogue'][0]['transcript'] for e in data]
            eng = re.compile(r'[\u0900-\u097F]')
            self.labels = [[l[1] for l in e['dialogue'][0]['turn_label'] if eng.match(l[1])] for e in data]
            


        self.sentence_count = len(self.sentences)
        self.token_ids = []
        for s in self.sentences:
            input_encoded = self.tokenizer.encode_plus(s, return_tensors="pt")
            # print([self.sentences[0], self.sentences[1]], input_encoded)
            # exit()
            # with torch.no_grad():
            #     input_encoded = input_encoded.to(self.device)
            #     states = self.model(**input_encoded).hidden_states
            # output = torch.stack([states[i] for i in range(len(states))])
            # output = output.squeeze()
            # output = torch.mean(output, dim=0)
            # output = output.to(torch.device('cpu'))
            
            # self.embeddings.append(output)
            self.token_ids.append(input_encoded["input_ids"].squeeze())
        
        if mlb == None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(self.labels)
            file = open(path.split(".")[0] + "_classes.json", "w+")
            json.dump({"classes": list(self.mlb.classes_)}, file, indent=4)
            file.close()
        else:
            self.mlb = mlb
        
        self.labels = self.mlb.transform(self.labels)
        
        # self.embedding_size = 300 + len(self.special_tokens)

    def __getitem__(self, index, return_sentence = False, pad = False):
        sentence = self.sentences[index][:]
        token_ids = self.token_ids[index]
        # add padding
        if pad:
            for _ in range(self.seq_len - len(sentence)):
                sentence.append("<PAD>")

        target = torch.tensor(self.labels[index], dtype=torch.float)

        if return_sentence:
            return sentence, token_ids, target
        return token_ids, target

    def __len__(self):
        return self.sentence_count
    
if __name__ == "__main__":
    dataset = BTP(path="data/HDRS_Corpus/final_train_hin.json")
    print(len(dataset), len(dataset.mlb.classes_))
    print(dataset[0])
    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset.pad_token_id)
    # dataset = BTP(path="BTP_data/atis/dev.txt")
    # print(len(dataset))
    # dataset = BTP(path="BTP_data/atis/test.txt")
    # print(len(dataset))

# SNIPS
# 13084 73
# 700
# 700

# ATIS
# 4478 121
# 500
# 893

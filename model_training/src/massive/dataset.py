import torch
from torch.utils.data import Dataset
import torchtext
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, AutoTokenizer
import json
from itertools import chain
from datasets import load_dataset
import pandas as pd

def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


class BTP(Dataset):

    # BERT Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # pad id
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    def __init__(self, split, slot_map = None, mode = "intent", mlb = None):

        self.mode = mode
        data = load_dataset("AmazonScience/massive", "en-US", split=split)
        # data = load_dataset("AmazonScience/massive", "hi-IN", split=split)
        self.sentences = []
        self.slots = []
        self.labels = []
        self.scenarios = []

        for entry in data:
            self.labels.append(entry["intent"])
            self.scenarios.append(entry["scenario"])
            sentence = entry["utt"].lower().split()
            # slots = [line.split(" ")[1] for line in temp[:-1]]

            # tokens beginning with [CLS]
            sentence_token_ids = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
            # sentence_slots = ["X"]
            for word in sentence:
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
                # token_slots = ["X"] * (len(token_ids) - 1)
                # token_slots.append(slot)
                sentence_token_ids.extend(token_ids)
                # sentence_slots.extend(token_slots)
            
            # tokens end with [SEP]
            sentence_token_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            # sentence_slots.append("X")

            self.sentences.append(sentence_token_ids)
            # self.slots.append(sentence_slots)


        self.sentence_count = len(self.sentences)
        self.seq_len = max([len(s)for s in self.sentences])
        self.avg_len = sum([len(s)for s in self.sentences]) / self.sentence_count
        self.intent_count = max(self.labels) + 1
        self.scenario_count = max(self.scenarios) + 1
        
        # if slot_map == None:
        #     self.slot_map = self.get_slot_map()
        # else:
        #     self.slot_map = slot_map
        
        # if mlb == None:
        #     self.mlb = MultiLabelBinarizer()
        #     self.mlb.fit(self.labels)
        #     file = open(path.split(".")[0] + "_classes.json", "w+")
        #     json.dump({"classes": list(self.mlb.classes_)}, file, indent=4)
        #     file.close()
        # else:
        #     self.mlb = mlb
        
        # self.labels = self.mlb.transform(self.labels)
        
        # self.embedding_size = 300 + len(self.special_tokens)

    def __getitem__(self, index, return_sentence = False, pad = False):
        sentence = self.sentences[index][:]
        # slots = self.slots[index][:]
        # add padding
        if pad:
            for _ in range(self.seq_len - len(sentence)):
                sentence.append("<PAD>")
                # slots.append("O")

        # slots = [self.slot_map[slot] if slot in self.slot_map else self.slot_map["<OOV>"] for slot in slots]

        if self.mode == "slot":
            # target = torch.tensor(slots)
            target = 1
        elif self.mode == "intent":
            target = torch.tensor(self.labels[index])

        if return_sentence:
            return sentence, torch.tensor(sentence, dtype=torch.long), target, torch.tensor(self.scenarios[index])
        return torch.tensor(sentence, dtype=torch.long), target, torch.tensor(self.scenarios[index])

    def __len__(self):
        return self.sentence_count
    
    def get_slot_map(self):
        map = {"<OOV>": 0}
        count = 1
        for sentence_slots in self.slots:
            for slot in sentence_slots:
                if slot not in map:
                    map[slot] = count
                    count += 1
        return map


class BTP2(Dataset):

    # BERT Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # pad id
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    def __init__(self, split, slot_map = None, mode = "intent", mlb = None):

        self.mode = mode
        data = pd.read_json("test_small.json")
        # data = load_dataset("AmazonScience/massive", "hi-IN", split=split)
        self.sentences = []
        self.slots = []
        self.labels = []
        self.scenarios = []

        for key, entry in data.items():
            self.labels.append(entry["intent"])
            self.scenarios.append(entry["scenario"])
            sentence = entry["utt"].lower().split()
            # slots = [line.split(" ")[1] for line in temp[:-1]]

            # tokens beginning with [CLS]
            sentence_token_ids = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
            # sentence_slots = ["X"]
            for word in sentence:
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
                # token_slots = ["X"] * (len(token_ids) - 1)
                # token_slots.append(slot)
                sentence_token_ids.extend(token_ids)
                # sentence_slots.extend(token_slots)
            
            # tokens end with [SEP]
            sentence_token_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            # sentence_slots.append("X")

            self.sentences.append(sentence_token_ids)
            # self.slots.append(sentence_slots)


        self.sentence_count = len(self.sentences)
        self.seq_len = max([len(s)for s in self.sentences])
        self.avg_len = sum([len(s)for s in self.sentences]) / self.sentence_count
        self.intent_count = max(self.labels) + 1
        self.scenario_count = max(self.scenarios) + 1
        
        # if slot_map == None:
        #     self.slot_map = self.get_slot_map()
        # else:
        #     self.slot_map = slot_map
        
        # if mlb == None:
        #     self.mlb = MultiLabelBinarizer()
        #     self.mlb.fit(self.labels)
        #     file = open(path.split(".")[0] + "_classes.json", "w+")
        #     json.dump({"classes": list(self.mlb.classes_)}, file, indent=4)
        #     file.close()
        # else:
        #     self.mlb = mlb
        
        # self.labels = self.mlb.transform(self.labels)
        
        # self.embedding_size = 300 + len(self.special_tokens)

    def __getitem__(self, index, return_sentence = False, pad = False):
        sentence = self.sentences[index][:]
        # slots = self.slots[index][:]
        # add padding
        if pad:
            for _ in range(self.seq_len - len(sentence)):
                sentence.append("<PAD>")
                # slots.append("O")

        # slots = [self.slot_map[slot] if slot in self.slot_map else self.slot_map["<OOV>"] for slot in slots]

        if self.mode == "slot":
            # target = torch.tensor(slots)
            target = 1
        elif self.mode == "intent":
            target = torch.tensor(self.labels[index])

        if return_sentence:
            return sentence, torch.tensor(sentence, dtype=torch.long), target, torch.tensor(self.scenarios[index])
        return torch.tensor(sentence, dtype=torch.long), target, torch.tensor(self.scenarios[index])

    def __len__(self):
        return self.sentence_count
    
    def get_slot_map(self):
        map = {"<OOV>": 0}
        count = 1
        for sentence_slots in self.slots:
            for slot in sentence_slots:
                if slot not in map:
                    map[slot] = count
                    count += 1
        return map

if __name__ == "__main__":
    dataset = BTP(split="train")
    print(len(dataset))
    print(dataset[0])
    print(dataset[0][0])
    print(dataset[0][1])
    print(dataset[0][2])
    print(dataset.seq_len)
    print(dataset.avg_len)
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

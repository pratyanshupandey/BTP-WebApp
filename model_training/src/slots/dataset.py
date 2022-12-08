import torch
from torch.utils.data import Dataset
import torchtext
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer
import json
from itertools import chain


def flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


class BTP(Dataset):

    # BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # pad id
    pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    def __init__(self, path, slot_map = None, mode = "slot", mlb = None):

        self.mode = mode

        # Get the sentences from the file
        file = open(path)
        corpus = file.read()
        file.close()

        data = corpus.split("\n\n")[:-1]
        self.sentences = []
        self.slots = []
        self.labels = []

        for i, entry in enumerate(data):
            temp = entry.split("\n")
            self.labels.append(temp[-1].split("#"))
            sentence = [line.split(" ")[0] for line in temp[:-1]]
            slots = [line.split(" ")[1] for line in temp[:-1]]
            if i == 8426:
                print(temp)
                print(sentence)
                print(slots)

            # tokens beginning with [CLS]
            sentence_token_ids = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
            sentence_slots = ["X"]
            for word, slot in zip(sentence, slots):
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word))
                token_slots = ["X"] * (len(token_ids) - 1)
                token_slots.append(slot)
                if len(token_ids) != len(token_slots):
                    print(i, token_ids, token_slots)
                sentence_token_ids.extend(token_ids)
                sentence_slots.extend(token_slots)
            
            # tokens end with [SEP]
            sentence_token_ids.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
            sentence_slots.append("X")

            self.sentences.append(sentence_token_ids)
            self.slots.append(sentence_slots)


        self.sentence_count = len(self.sentences)
        self.seq_len = max([len(s)for s in self.sentences])
        self.avg_len = sum([len(s)for s in self.sentences]) / self.sentence_count
        
        if slot_map == None:
            self.slot_map = self.get_slot_map()
        else:
            self.slot_map = slot_map
        
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
        slots = self.slots[index][:]
        # add padding
        if pad:
            for _ in range(self.seq_len - len(sentence)):
                sentence.append("<PAD>")
                slots.append("O")

        slots = [self.slot_map[slot] if slot in self.slot_map else self.slot_map["<OOV>"] for slot in slots]

        if self.mode == "slot":
            target = torch.tensor(slots)
        elif self.mode == "intent":
            target = torch.tensor(self.labels[index], dtype=torch.float)

        length = sum(target != self.slot_map["X"])

        if return_sentence:
            return sentence, torch.tensor(sentence, dtype=torch.long), target, length
        return torch.tensor(sentence, dtype=torch.long), target, length

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
    dataset = BTP(path="BTP_data/snips/train.txt")
    print(len(dataset), len(dataset.slot_map), len(dataset.mlb.classes_))
    print(dataset.__getitem__(8426, True))
    print(dataset.seq_len)
    print(dataset.avg_len)
    for i in range(len(dataset)):
        try:
            assert dataset[i][0].size() == dataset[i][1].size()
        except Exception as e:
            print(i)
    # print(dataset.slot_map)
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

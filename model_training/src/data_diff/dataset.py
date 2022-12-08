import torch
from torch.utils.data import Dataset
import torchtext
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

class BTP(Dataset):

    # special tokens
    special_tokens = {
        "<OOV>" : torch.tensor([0, 1]),
        "<PAD>" : torch.tensor([1, 0]),
    }

    # embedding
    glove = torchtext.vocab.GloVe(name='6B', dim=300)

    def __init__(self, path, slot_map = None, mode = "none", mlb = None):

        self.mode = mode

        # Get the sentences from the file
        file = open(path)
        corpus = file.read()
        file.close()

        data = corpus.split("\n\n")[:-1]
        self.sentences = []
        self.slots = []
        self.labels = []

        for entry in data:
            temp = entry.split("\n")
            self.labels.append(temp[-1].split("#"))
            self.sentences.append([line.split(" ")[0] for line in temp[:-1]])
            self.slots.append([line.split(" ")[1] for line in temp[:-1]])

        self.sentence_count = len(self.sentences)
        self.seq_len = 46 #max([len(s)for s in self.sentences])
        self.vocab = self.get_embeddings()
        
        if slot_map == None:
            self.slot_map = self.get_slot_map()
        else:
            self.slot_map = slot_map
        
        if mlb == None:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(self.labels)
        else:
            self.mlb = mlb
        
        self.labels = self.mlb.transform(self.labels)
        
        self.embedding_size = 300 + len(self.special_tokens)

    def __getitem__(self, index, return_sentence = False, pad = True):
        sentence = self.sentences[index][:]
        slots = self.slots[index][:]
        # add padding
        if pad:
            for _ in range(self.seq_len - len(sentence)):
                sentence.append("<PAD>")
                slots.append("O")

        tokens = [word if word in self.vocab else "<OOV>" for word in sentence]
        embeddings = [self.vocab[word]["emb"] for word in tokens]
        slots = [self.slot_map[slot] if slot in self.slot_map else self.slot_map["<OOV>"] for slot in slots]

        if self.mode == "slot":
            target = torch.tensor(slots)
        elif self.mode == "intent":
            target = torch.tensor(self.labels[index], dtype=torch.float)

        if return_sentence:
            return sentence, torch.stack(embeddings), target
        if self.mode == "none":
            return torch.stack(embeddings)
        else:
            return torch.stack(embeddings), target

    def __len__(self):
        return self.sentence_count
    

    def get_embeddings(self):
        vocab = defaultdict(lambda: 0)
        for sentence in self.sentences:
            for word in sentence:
                vocab[word] += 1
        
        # print(sum([vocab[word] == 1 for word in vocab.keys()]))
        # print(sum([vocab[word] == 2 for word in vocab.keys()]))
        # print(sum([vocab[word] == 3 for word in vocab.keys()]))

        unkown_count = 0
        vocab_idx = 0
        reduced_vocab = {}

        # add special tokens
        for key, val in self.special_tokens.items():
            reduced_vocab[key] = {
                "emb" : torch.cat([val, torch.zeros(300)]),
                "idx" : vocab_idx
            }
            vocab_idx += 1

        # Add the rest of the words
        for word in vocab.keys():
            # Not converted to lower case so lower case backup is needed
            emb = self.glove.get_vecs_by_tokens([word], lower_case_backup=True)[0]
            if torch.count_nonzero(emb) == 0:
                unkown_count += 1
            else:
                reduced_vocab[word] = {
                    "emb" : torch.cat([torch.zeros(len(self.special_tokens)), emb]), 
                    "idx": vocab_idx
                }
                vocab_idx += 1

        # print(f"{unkown_count} unknown words found")

        return reduced_vocab
        
    def get_slot_map(self):
        map = {"<OOV>": 0}
        count = 1
        for sentence_slots in self.slots:
            for slot in sentence_slots:
                if slot not in map:
                    map[slot] = count
                    count += 1
        return map



class CombinedDataset(Dataset):
    def __init__(self, atis: BTP, snips: BTP) -> None:
        super().__init__()
        self.atis = atis
        self.snips = snips
        self.length = atis.sentence_count + snips.sentence_count
    
    def __getitem__(self, index):
        if index < self.atis.sentence_count:
            return self.atis[index], 0
        else:
            return self.snips[index - self.atis.sentence_count], 1

    def __len__(self):
        return self.length


if __name__ == "__main__":
    dataset = BTP(path="BTP_data/atis/train.txt")
    print(len(dataset), len(dataset.slot_map), len(dataset.mlb.classes_))
    print(dataset[0][0].dtype)
    print(dataset[0][1].dtype)
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

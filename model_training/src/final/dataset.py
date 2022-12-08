from torch.utils.data import Dataset
from datasets import load_dataset
import json

class BTP(Dataset):
    def __init__(self, dataset_name, split, mlb = None):
        self.sentences = []
        self.labels = []

        if dataset_name == "massive":
            data = load_dataset("AmazonScience/massive", "en-US", split=split)
            for entry in data:
                self.labels.append(entry["intent"])
                self.sentences.append(entry["utt"].lower())

        elif dataset_name in ["atis", "snips"]:
            file = open("data/" +  dataset_name + "/" + split + ".json")
            data = json.load(file)
            file.close()

            self.sentences = data["sentences"]
            self.labels = data["labels"]
        
        else:
            print("Incorrect Dataset")
            exit(0)

        
        self.sentence_count = len(self.sentences)
        self.max_len = max([len(s.split())for s in self.sentences])
        self.avg_len = sum([len(s.split())for s in self.sentences]) / self.sentence_count
        self.intent_count = max(self.labels) + 1
        

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return self.sentence_count
    

if __name__ == "__main__":
    dataset = BTP(data="massive" ,split="train")
    print(len(dataset))
    print(dataset.avg_len)
    print(dataset.max_len)
    print(dataset[0])

# SNIPS
# 13084 73
# 700
# 700

# ATIS
# 4478 121
# 500
# 893

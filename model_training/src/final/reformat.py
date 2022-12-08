import argparse
import json
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--inp', default="", type=str)
parser.add_argument('--out', default="", type=str)
CONFIG = parser.parse_args()


def reformat(input, output, lbl_enc = None):
    file = open(input)
    corpus = file.read()
    file.close()

    data = corpus.split("\n\n")[:-1]
    sentences = []
    labels = []

    for entry in data:
        temp = entry.split("\n")
        
        tmp_labels = temp[-1].split("#")
        if len(tmp_labels) > 1:
            tmp_labels = list(filter(lambda item: item != "atis_flight", tmp_labels))
            if len(tmp_labels) > 1:
                tmp_labels = list(filter(lambda item: item != "atis_airline", tmp_labels))

        label = tmp_labels[0]

        if label == 'atis_day_name':
            label = "atis_flight"
        
        labels.append(label)
        sentence = " ".join([line.split(" ")[0] for line in temp[:-1]]).lower()
        sentences.append(sentence)

    assert len(sentences) == len(labels)

    if lbl_enc == None:
        lbl_enc = LabelEncoder()
        lbl_enc.fit(labels)

    labels = lbl_enc.transform(labels).tolist()

    file = open(output, "w+")
    json.dump({
            "sentences": sentences,
            "labels": labels
            }, 
            file)
    file.close()

    return lbl_enc


inp_dir = CONFIG.inp
out_dir = CONFIG.out

if inp_dir[-1] != "/":
    inp_dir += "/"

if out_dir[-1] != "/":
    out_dir += "/"

train_lbl_enc = reformat(inp_dir + "train.txt", out_dir + "train.json")
train_lbl_enc = reformat(inp_dir + "dev.txt", out_dir + "validation.json", train_lbl_enc)
train_lbl_enc = reformat(inp_dir + "test.txt", out_dir + "test.json", train_lbl_enc)


file = open(out_dir + "intent_classes.json", "w+")
json.dump({"classes": train_lbl_enc.classes_.tolist()}, file, indent=4)
file.close()
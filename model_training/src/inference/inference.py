import torch
import torch.nn as nn
from model import TorchGRU, TorchGRUIntent
from dataset import BTP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)


class Inference(nn.Module):
    def __init__(self) -> None:
        super(Inference, self).__init__()
                
        atis_intent = "../intent/checkpoints/TGRU_2b_atisfinal.pt"
        snips_intent = "../intent/checkpoints/TGRU_2b_snipsfinal.pt"
        atis_slots = "../slots/checkpoints/TGRU_2b_atisfinal.pt"
        snips_slots = "../slots/checkpoints/TGRU_2b_snipsfinal.pt"

        self.atis_intent_model = TorchGRUIntent(embedding_size=302, hidden_size=300, vocab_size=17)
        self.snips_intent_model = TorchGRUIntent(embedding_size=302, hidden_size=300, vocab_size=7)
        self.atis_slots_model = TorchGRU(embedding_size=302, hidden_size=300, vocab_size=121)
        self.snips_slots_model = TorchGRU(embedding_size=302, hidden_size=300, vocab_size=73)

        self.atis_intent_model.load_state_dict(torch.load(atis_intent))
        self.snips_intent_model.load_state_dict(torch.load(snips_intent))
        self.atis_slots_model.load_state_dict(torch.load(atis_slots))
        self.snips_slots_model.load_state_dict(torch.load(snips_slots))

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        atis_probs = self.sigmoid(self.atis_intent_model(x))
        snips_probs = self.sigmoid(self.snips_intent_model(x))

        if max(atis_probs) > max(snips_probs):
            # process slots as atis
            return "atis", self.atis_slots_model(x)
        else:
            return "snips", self.snips_slots_model(x)
    

# atis
data_dir = "BTP_data/atis/"
atis_train_dataset = BTP(data_dir + "train.txt")
atis_val_dataset = BTP(data_dir + "dev.txt", slot_map=atis_train_dataset.slot_map, mlb=atis_train_dataset.mlb)
atis_test_dataset = BTP(data_dir + "test.txt", slot_map=atis_train_dataset.slot_map, mlb=atis_train_dataset.mlb)


# snips
data_dir = "BTP_data/snips/"
snips_train_dataset = BTP(data_dir + "train.txt")
snips_val_dataset = BTP(data_dir + "dev.txt", slot_map=snips_train_dataset.slot_map, mlb=snips_train_dataset.mlb)
snips_test_dataset = BTP(data_dir + "test.txt", slot_map=snips_train_dataset.slot_map, mlb=snips_train_dataset.mlb)


inference = Inference().to(device)

correct_set_prediction = 0
correct_slots = 0
total_slots = 0

softmax = nn.Softmax(dim = 0)

def gen_accuracy_scores(dataset : BTP, type: str):
    global correct_set_prediction
    global correct_slots
    global total_slots

    with torch.no_grad():
        inference.eval()
        for i in range(dataset.sentence_count):
            embeddings, slots, labels = dataset.__getitem__(i, return_sentence=False, pad=False)
            
            embeddings = embeddings.to(device)
            slots = slots.to(device)

            pred_set, slot_probs = inference(embeddings)
            if pred_set == type:
                correct_set_prediction += 1
                for j in range(len(slots)):
                    pred_slot = slot_probs[j]
                    idx = slots[j]

                    # Applying Softmax
                    pred_slot = softmax(pred_slot)

                    if torch.argmax(pred_slot) == idx:
                        correct_slots += 1
                total_slots += len(slots)

    
gen_accuracy_scores(atis_test_dataset, "atis")
gen_accuracy_scores(snips_test_dataset, "snips")
print("Correct Set Classification: ", correct_set_prediction / (atis_test_dataset.sentence_count + snips_test_dataset.sentence_count))
print("Slot Accuracy: ", correct_slots / total_slots)
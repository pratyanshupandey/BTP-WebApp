import torch 
from torch import nn
from model import TorchGRU, TorchLSTM
from utils import Trainer, EarlyStopping
from dataset import BTP
import argparse

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--cpu_cores', default=2, type=int)
parser.add_argument('--data', default="", type=str)
parser.add_argument('--model_type', default="RNN", type=str)
parser.add_argument('--model', default="", type=str)
parser.add_argument('--mode', default="train", type=str)
CONFIG = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

checkpoint_dir = "checkpoints/"

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 300
cpu_cores = CONFIG.cpu_cores
print(f"Using {cpu_cores} CPU cores")

# Getting dataset and data loaders
data_dir = CONFIG.data
if data_dir[-1] != "/":
    data_dir += "/"

# Getting train and dataset and using train dataset's vocabulary for val and test
train_dataset = BTP(data_dir + "train.txt")
val_dataset = BTP(data_dir + "dev.txt", slot_map=train_dataset.slot_map)
test_dataset = BTP(data_dir + "test.txt", slot_map=train_dataset.slot_map)
pad_index = train_dataset.slot_map["X"]

def batch_sequences(seq_list):
    token_ids = []
    slots = [] 
    length_sum = 0
    for ids, target, length in seq_list:
        token_ids.append(ids)
        slots.append(target)
        length_sum += length

    token_ids = nn.utils.rnn.pad_sequence(token_ids, padding_value=BTP.pad_token_id, batch_first=True)
    slots = nn.utils.rnn.pad_sequence(slots, padding_value=pad_index, batch_first=True)
    attention_mask = (token_ids != BTP.pad_token_id)
    return token_ids, attention_mask, slots, length_sum

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

# Getting the model
if CONFIG.model_type == "TGRU":
    model = TorchGRU(hidden_size=300, vocab_size=len(train_dataset.slot_map))
elif CONFIG.model_type == "TLSTM":
    model = TorchLSTM(hidden_size=300, vocab_size=len(train_dataset.slot_map))
else:
    print("Unidentified model type")
    exit(1)

if CONFIG.model == "":
    print("Training new model ", type(model))
else:
    print("Using model from", CONFIG.model)
    model.load_state_dict(torch.load(CONFIG.model))
    model = model.to(device)


# Optimizer and Criterion
criterion = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min', 
                                                        factor=0.5, 
                                                        patience=4, 
                                                        verbose=True)  


# Early Stopping
early_stopping = EarlyStopping(patience=9)

# Train the model
trainer = Trainer(model_name="BERT_TGRU_2b_atis",
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    early_stopping=early_stopping,
                    log_periodicity=100,
                    padding_index=pad_index,
                    checkpoint_strategy="both",
                    checkpoint_periodicity=10)

if CONFIG.mode == "train":
    trainer.train()

# Test
trainer.evaluate(name="Train", loader=train_loader)
trainer.evaluate(name="Val", loader=val_loader)
trainer.evaluate(name="Test", loader=test_loader)

# Generate accuracy scores
def gen_accuracy_scores(dataset : BTP, trainer : Trainer, name: str):
    correct_sum = 0
    total_sum = 0

    for i in range(dataset.sentence_count):
        ids, slots, length = dataset.__getitem__(i, return_sentence=False, pad=False)
        mask = (ids != BTP.pad_token_id)
        try:
            correct = trainer.get_seq_accuracy(ids, mask, slots)
        except Exception as e:
            print(i)
        correct_sum += correct
        total_sum += length
    
    print(f'{name} - Accuracy: {correct_sum / total_sum}')

gen_accuracy_scores(train_dataset, trainer, "Train")
gen_accuracy_scores(val_dataset, trainer, "Val")
gen_accuracy_scores(test_dataset, trainer, "Test")

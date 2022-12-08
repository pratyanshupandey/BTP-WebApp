import torch 
from torch import nn
from model import IntentModel
from utils import Trainer, EarlyStopping
from dataset import BTP
import argparse
from transformers import BertTokenizer


parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--cpu_cores', default=10, type=int)
parser.add_argument('--data', default="massive", type=str)
parser.add_argument('--model', default="", type=str)
parser.add_argument('--model_name', default="model", type=str)
parser.add_argument('--mode', default="train", type=str)
CONFIG = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

checkpoint_dir = "checkpoints/"

# Hyperparameters
batch_size = 128
learning_rate = 1e-4
epochs = 300
cpu_cores = CONFIG.cpu_cores
max_len = 32
print(f"Using {cpu_cores} CPU cores")

# Getting dataset and data loaders
data = CONFIG.data

train_dataset = BTP(data, "train")
val_dataset = BTP(data, "validation")
test_dataset = BTP(data, "test")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def batch_sequences(seq_list):
    sentences = []
    labels = []
    for (sent, label) in seq_list:
        sentences.append(sent)
        labels.append(label)

    sents_bert_toks = tokenizer.batch_encode_plus(sentences, 
                                                padding="longest", 
                                                max_length=max_len, 
                                                truncation=True, 
                                                return_tensors="pt")
    return sents_bert_toks, torch.tensor(labels)



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

model = IntentModel(labels_count=train_dataset.intent_count)

if CONFIG.model == "":
    print("Training new model ", type(model))
else:
    print("Using model from", CONFIG.model)
    model.load_state_dict(torch.load(CONFIG.model))
    model = model.to(device)


# Optimizer and Criterion
# weights = [1] * train_dataset.intent_count
# if data == "atis":
#     weights[9] = 0.1 # atis_flight
# weights = torch.tensor(weights).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min', 
                                                        factor=0.5, 
                                                        patience=4, 
                                                        verbose=True)  


# Early Stopping
early_stopping = EarlyStopping(patience=9)

# Train the model
trainer = Trainer(model_name=CONFIG.model_name,
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
                    log_periodicity=10,
                    checkpoint_strategy="periodic",
                    checkpoint_periodicity=1,
                    test_loader=test_loader)

if CONFIG.mode == "train":
    trainer.train()

# Test
trainer.evaluate(name="Val", loader=val_loader)
trainer.evaluate(name="Test", loader=test_loader)

import torch 
from torch import nn
from model import TorchGRUDiff, TorchLinearDiff
from utils import Trainer, EarlyStopping
from dataset import BTP, CombinedDataset
import argparse

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--cpu_cores', default=2, type=int)
parser.add_argument('--data', default="", type=str)
parser.add_argument('--model_type', default="Linear", type=str)
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
main_data_dir = CONFIG.data
if main_data_dir[-1] != "/":
    main_data_dir += "/"

# atis
# atis
data_dir = main_data_dir + "atis/"
atis_train_dataset = BTP(data_dir + "train.txt")
atis_val_dataset = BTP(data_dir + "dev.txt", slot_map=atis_train_dataset.slot_map, mlb=atis_train_dataset.mlb)
atis_test_dataset = BTP(data_dir + "test.txt", slot_map=atis_train_dataset.slot_map, mlb=atis_train_dataset.mlb)


# snips
data_dir = main_data_dir + "snips/"
snips_train_dataset = BTP(data_dir + "train.txt")
snips_val_dataset = BTP(data_dir + "dev.txt", slot_map=snips_train_dataset.slot_map, mlb=snips_train_dataset.mlb)
snips_test_dataset = BTP(data_dir + "test.txt", slot_map=snips_train_dataset.slot_map, mlb=snips_train_dataset.mlb)

train_dataset = CombinedDataset(atis_train_dataset, snips_train_dataset)
val_dataset = CombinedDataset(atis_val_dataset, snips_val_dataset)
test_dataset = CombinedDataset(atis_test_dataset, snips_test_dataset)


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            num_workers=cpu_cores)

val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores)

# Getting the model
if CONFIG.model_type == "TGRU":
    model = TorchGRUDiff(embedding_size=train_dataset.atis.embedding_size, hidden_size=300, vocab_size=2)
elif CONFIG.model_type == "Linear":
    model = TorchLinearDiff(embedding_size=train_dataset.atis.embedding_size, hidden_size=300, vocab_size=2)
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
trainer = Trainer(model_name="TGRU_2b",
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    early_stopping=early_stopping)

if CONFIG.mode == "train":
    trainer.train()

# Test
trainer.evaluate(name="Val", loader=val_loader)
trainer.evaluate(name="Test", loader=test_loader)

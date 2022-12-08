import torch
from os.path import exists

class Trainer:
    
    def __init__(self, model_name, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, checkpoint_dir, early_stopping, log_periodicity, **kwargs):
        self.model_name = model_name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        if self.checkpoint_dir[-1] != '/':
            self.checkpoint_dir += '/'
        self.log_periodicity = log_periodicity
        self.kwargs = kwargs
        self.softmax0 = torch.nn.Softmax(dim = 0)
        self.softmax1 = torch.nn.Softmax(dim = 1)
        

    def train(self):
        print(f"Started Training of {self.model_name}")
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train for an epoch
            for i, (ids, mask, slots, length_sum) in enumerate(self.train_loader):
                ids = ids.to(self.device)     # shape B * seq_len * E
                mask = mask.to(self.device)     # shape B * seq_len
                slots = slots.to(self.device) # shape B * seq_len

                slots = torch.transpose(slots, 0, 1)

                # shape seq_len * B * vocab
                vocab_probs = self.model(ids, mask)
                seq_len = len(slots)

                loss = 0
                for j in range(seq_len):
                    loss += self.criterion(vocab_probs[j], slots[j])
                
                loss /= length_sum

                # Update the weights
                self.optimizer.zero_grad()
                running_loss += loss
                loss.backward()
                self.optimizer.step()

                if i % self.log_periodicity == (self.log_periodicity - 1):
                    print('[%d, %d] loss: %.6f' %
                          (epoch + 1, i + 1, running_loss / self.log_periodicity), flush=True)
                    running_loss = 0.0

            # Calculate validation loss
            val_loss, accuracy = self.calculate_loss_accuracy(loader=self.val_loader)
            print('Epoch: %d Validation loss: %.5f accuracy: %.5f' % (epoch + 1, val_loss, accuracy), flush=True)

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Take a early stopping step
            self.early_stopping.step(val_loss)
            
            # Save a checkpoint according to strategy
            self.checkpoint(epoch)

            # Check early stopping to finish training
            if self.early_stopping.stop_training:
                print("Early Stopping the training")
                break

            # check external instructions to stop training
            if exists("STOP"):
                print("External instruction to stop the training")
                break

        print('Finished Training')
        self.save_model(self.model_name + "final")

    def calculate_loss_accuracy(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Test validation data
        with torch.no_grad():
            for i, (ids, mask, slots, length_sum) in enumerate(self.train_loader):
                ids = ids.to(self.device)     # shape B * seq_len * E
                mask = mask.to(self.device)     # shape B * seq_len
                slots = slots.to(self.device) # shape B * seq_len

                slots = torch.transpose(slots, 0, 1)

                # shape seq_len * B * vocab
                vocab_probs = self.model(ids, mask)
                seq_len = len(slots)

                loss = 0
                for j in range(seq_len):
                    slot_probs = vocab_probs[j] # shape B * vocab
                    idxs = slots[j]      # shape B
                    loss += self.criterion(vocab_probs[j], slots[j])

                    # Applying Softmax
                    slot_probs = self.softmax1(slot_probs)

                    # Accuracy
                    for prob, idx in zip(slot_probs, idxs):
                        if idx != self.kwargs["padding_index"]:
                            if torch.argmax(prob) == idx:
                                correct += 1
                            total += 1

                loss /= length_sum
                total_loss += loss.data
                
        self.model.train()
        return total_loss / len(loader), correct / total


    def get_seq_accuracy(self, ids, mask, slots):
        self.model.eval()
        correct = 0

        with torch.no_grad():
            ids = ids.to(self.device)         # shape seq_len * E
            mask = mask.to(self.device)         # shape seq_len
            slots = slots.to(self.device)     # shape seq_len

            seq_len = len(slots)

            vocab_prob = self.model(ids.view(1,-1), mask.view(1,-1)) # shape seq_len * 1 * vocab
            for j in range(seq_len):
                slot_prob = vocab_prob[j][0]
                idx = slots[j]

                # Applying Softmax
                slot_prob = self.softmax0(slot_prob)

                if idx != self.kwargs["padding_index"] and torch.argmax(slot_prob) == idx:
                    correct += 1
        return correct

    def save_model(self, file_name):
        print(f'Saving Model in {self.checkpoint_dir + file_name}')
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir + file_name}.pt")

    def evaluate(self, name, loader):
        loss, accuracy = self.calculate_loss_accuracy(loader)
        print(f"{name} loss = {loss} \t {name} accuracy = {accuracy}")

    def checkpoint(self, epoch):
        save_checkpoint = False
        checkpoint_name = ""
        if self.kwargs["checkpoint_strategy"] == "periodic" and epoch % self.kwargs["checkpoint_periodicity"] == (self.kwargs["checkpoint_periodicity"] - 1):
            save_checkpoint = True
            checkpoint_name = f"checkpoint_{epoch}"
        elif self.kwargs["checkpoint_strategy"] == "best" and self.early_stopping.current_count == 0:
            save_checkpoint = True
            checkpoint_name = "checkpoint_best"
        elif self.kwargs["checkpoint_strategy"] == "both":
            if self.early_stopping.current_count == 0:
                save_checkpoint = True
                checkpoint_name = "checkpoint_best"
            elif epoch % self.kwargs["checkpoint_periodicity"] == (self.kwargs["checkpoint_periodicity"] - 1):
                save_checkpoint = True
                checkpoint_name = f"checkpoint_{epoch}"
            
        if save_checkpoint:
            self.save_model(self.model_name + checkpoint_name)
        

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = 1e5
        self.current_count = 0
        self.stop_training = False
    
    def step(self, val_loss):
        if val_loss < self.min_loss:
            self.current_count = 0
            self.min_loss = val_loss
        else:
            self.current_count += 1
        
        if self.current_count >= self.patience:
            self.stop_training = True

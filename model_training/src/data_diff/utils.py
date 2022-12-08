import torch

class Trainer:
    
    def __init__(self, model_name, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, checkpoint_dir, early_stopping):
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
        self.softmax0 = torch.nn.Softmax(dim=0)
        self.softmax1 = torch.nn.Softmax(dim=1)
        

    def train(self):
        print(f"Started Training of {self.model_name}")
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Train for an epoch
            for i, (embeddings, labels) in enumerate(self.train_loader):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # make embediings from B*seq_len*emb_size -> seq_len*B*emb_size
                # labels  B* (#classes)
                embeddings = torch.transpose(embeddings, 0, 1)
                loss = 0
                out = self.model(embeddings) # B * (#classes)
                loss = self.criterion(out, labels)
                
                # Update the weights
                self.optimizer.zero_grad()
                running_loss += loss
                loss.backward()
                self.optimizer.step()

                if i % 100 == 99:
                    print('[%d, %d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 100), flush=True)
                    running_loss = 0.0

            # Calculate validation loss
            val_loss, accuracy = self.calculate_loss_accuracy(loader=self.val_loader)
            print('Epoch: %d Validation loss: %.5f accuracy: %.5f' % (epoch + 1, val_loss, accuracy), flush=True)

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Take a early stopping step
            self.early_stopping.step(val_loss)
            
            # save a model every 10 epochs and also calculate the accuracy
            # if epoch % 10 == 9:
            #     self.save_model(self.model_name + f"checkpoint_{epoch}")
            
            # Check early stopping to finish training
            if self.early_stopping.stop_training:
                print("Early Stopping the training")
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
            for i, (embeddings, labels) in enumerate(loader):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                # make embediings from B*seq_len*emb_size -> seq_len*B*emb_size
                # make indcies from B*seq_len -> seq_len*B
                embeddings = torch.transpose(embeddings, 0, 1)
                loss = 0
                pred_classes = self.model(embeddings) # B * (#classes)
                loss = self.criterion(pred_classes, labels)

                # Applying sigmoid
                pred_classes = self.softmax1(pred_classes)

                # Accumulating loss, accuracy
                total_loss += loss.data
                correct += sum(torch.argmax(pred_classes, dim = 1) == labels)
                total += len(labels)


        self.model.train()
        return total_loss / len(loader), correct / total

    def save_model(self, file_name):
        print(f'Saving Model in {self.checkpoint_dir + file_name}')
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir + file_name}.pt")

    def evaluate(self, name, loader):
        loss, accuracy = self.calculate_loss_accuracy(loader)
        print(f"{name} loss = {loss} \t {name} accuracy = {accuracy}")


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

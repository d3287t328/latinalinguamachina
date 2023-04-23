# This script assumes that you have already preprocessed your dataset and implemented your own dataset class (MyDataset) and model class (MyModel) accordingly. 
# It also uses torch.optim.lr_scheduler.ReduceLROnPlateau to reduce the learning rate if the validation loss does not improve for a certain number of epochs 
# (patience). Finally, it uses an EarlyStopping helper class to stop training early if the validation loss does not improve for a certain number of epochs 
# (patience) and to save the best model parameters based on the validation loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from my_dataset import MyDataset  # replace with your own dataset class
from my_model import MyModel  # replace with your own model class

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define hyperparameters
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
num_epochs = 10
max_grad_norm = 1.0
patience = 3
factor = 0.5
min_lr = 1e-6

# create dataset and data loaders
train_dataset = MyDataset(train=True)  # replace with your own train dataset instance
val_dataset = MyDataset(train=False)  # replace with your own validation dataset instance
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# create model and optimizer
model = MyModel().to(device)  # replace with your own model instance
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# define loss function
criterion = nn.CrossEntropyLoss()

# define learning rate scheduler and early stopping
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr)
early_stopper = EarlyStopping(patience=patience, verbose=True)

# training loop
for epoch in range(num_epochs):
    # train
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # clip gradients
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
        val_loss /= len(val_loader)

    # update learning rate scheduler and early stopper
    scheduler.step(val_loss)
    early_stop = early_stopper.step(val_loss, model)
    if early_stop:
        break

    # print progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# save model
torch.save(model.state_dict(), 'my_model.pt')  # replace with your desired file path

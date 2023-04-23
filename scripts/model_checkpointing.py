# In this script, the save_checkpoint function takes as input the model, optimizer, and a filename for the checkpoint file, and saves the current state of these objects to the file using the torch.save function. The load_checkpoint function takes as input the model, optimizer, and the same filename, and loads the state of these objects from the file using the torch.load function.
# In the training loop, the save_checkpoint function is called after every epoch to save a checkpoint of the current state of the model and optimizer. After training is complete, the load_checkpoint function can be called to load the latest checkpoint and resume training if desired.
# Note that this is just an example, and you may need to modify the script to suit your specific use case. For example, you may want to include additional information in the checkpoint file, such as the current epoch or batch index, or you may want to save multiple checkpoints at different intervals.


import torch
import os

def save_checkpoint(model, optimizer, filename):
    checkpoint_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint_dict, filename)
    print(f"Saved checkpoint as {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.exists(filename):
        checkpoint_dict = torch.load(filename)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")
    else:
        print(f"No checkpoint found at {filename}")

# Example usage
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint_filename = 'model_checkpoint.pth'

# Training loop
for epoch in range(num_epochs):
    # Train the model for one epoch
    train_loss = train(model, optimizer, train_loader)

    # Save a checkpoint after every epoch
    save_checkpoint(model, optimizer, checkpoint_filename)

# After training is complete, load the latest checkpoint and resume training if desired
load_checkpoint(model, optimizer, checkpoint_filename)
for epoch in range(resume_epoch, num_epochs):
# Train the model for one epoch
    train_loss = train(model, optimizer, train_loader)

    # Save a checkpoint after every epoch
    save_checkpoint(model, optimizer, checkpoint_filename)
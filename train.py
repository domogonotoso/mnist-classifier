# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MnistCNN

# Set computation device
# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Normalize MNIST dataset using its mean and std
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

# Load training and test datasets from torchvision with normalization applied
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)


# Split data into batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




# Initialize model, loss function, and optimizer
model = MnistCNN().to(device)  # Move model to GPU if available
criterion = nn.CrossEntropyLoss() # For classification, use Cross Enropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start train
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)  # Move input data to the same device as model

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    
torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model saved to mnist_cnn.pth")



from model import MnistCNN
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


batch_size = 64

# Load model
model = MnistCNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval() # Evaluation mode(No backprop, Dropout disabled)


# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# evaluation
correct = 0
total = 0
with torch.no_grad(): # No need to track gradients
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1) # get the index of the max probability. (CrossEntropy includes softmax)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
print(f"Test Accuracy: {100 * correct / total:.2f}%")

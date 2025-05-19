from model import MnistCNN
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


batch_size = 64

# Load model
model = MnistCNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()


#asdf
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) 
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


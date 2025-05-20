# MNIST Classifier

I studied AI from before, but I need to practice how to use git and organize its repository. And MNIST classifier is simple ML model with which practice git.

## What is it?

Implementing a CNN to classify 28x28 handwritten digits.

## Setup

While setting up a new conda environment, I needed to install the required libraries using:

```bash
pip install torch torchvision
```

## Error encountered during training

```text
ValueError: Expected input batch_size (256) to match target batch_size (64)
```

I misunderstood how the shape of the data changes through the CNN layers.

I assumed that the output would match the expected shape without carefully tracking the intermediate feature map sizes.

In `model.py`, I wrote:

```python
def forward(self, x):
    x = F.relu(self.conv1(x))                  # (batch, 32, 28, 28)
    x = self.pool(F.relu(self.conv2(x)))       # (batch, 64, 14, 14)
    x = self.dropout(x)
    print("Before flatten:", x.shape)
    x = x.view(-1, 64 * 7 * 7)                 # Flatten: reshape to (batch, 64 * 7 * 7) for FC layer
    print("After flatten:", x.shape)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)                            # Softmax will be applied later
    return x
```

The output showed:

```text
Before flatten: torch.Size([64, 64, 14, 14])
After flatten: torch.Size([256, 3136])
```

I realized that even though I understood the theory of CNNs, I had overlooked how tensor shapes change through pooling layers.
This led to a batch size mismatch error. After visualizing tensor shapes with print statements, I learned how important it is to trace dimensions carefully in real implementation.

## Training Log

```text
Epoch [1/5], Loss: 0.1748  
Epoch [2/5], Loss: 0.0633  
Epoch [3/5], Loss: 0.0475  
Epoch [4/5], Loss: 0.0410  
Epoch [5/5], Loss: 0.0364
```

The model trained well and achieved good convergence.




### Revision (19/05)
Erase test data at train.py
Show accuracy at train process



## Test

```text
Test Accuracy: 99.18%
```

Great!! But I need to know accruacy at each train loop

## After revision
'''text
Epoch [1/5], Loss: 0.1750, Accuracy: 94.50%
Epoch [2/5], Loss: 0.0646, Accuracy: 98.02%
Epoch [3/5], Loss: 0.0496, Accuracy: 98.42%
Epoch [4/5], Loss: 0.0402, Accuracy: 98.69%
Epoch [5/5], Loss: 0.0351, Accuracy: 98.89%
Model saved to mnist_cnn.pth

Test Accuracy: 99.15%
```

Great!!
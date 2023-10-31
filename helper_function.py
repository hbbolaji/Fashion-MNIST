import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_step import test_step

# Hyperparameters
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def accuracy(logits:torch.Tensor, labels:torch.Tensor):
  prediction = torch.argmax(logits, dim=1)
  correct = (prediction == labels).sum()
  acc = 100 * correct / len(labels)
  return acc

def eval_model(model:nn.Module, data_loader: DataLoader, criterion: nn.Module, accuracy):
  model.eval()
  with torch.inference_mode():
    acc, loss = test_step(model=model, data_loader=data_loader, criterion=criterion, accuracy=accuracy, device= 'cpu' or 'cuda')
    return {
      "model_name": model.__class__.__name__,
      "model_loss": loss,
      "model_accuracy": acc
    }

def visualize(images:torch.Tensor, labels:torch.Tensor):
  for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].view(28, 28, 1), cmap='gray')
    plt.title(classes[labels[i]])
  return
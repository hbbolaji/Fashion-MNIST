import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_step(model: nn.Module, data_loader:DataLoader, criterion:nn.Module, accuracy, optimizer:torch.optim.SGD, device: torch.device):
  """
  Args: 
    model: nn.Module
    data_loader:DataLoader
    criterion:nn.Module
    accuracy: function
    optimizer:torch.optim.SGD
    device: torch.device
  Returns:
    Nothing
  """
  model.train()
  train_acc, train_loss = 0, 0
  for batch, (images, labels) in enumerate(data_loader):
    images = images.to(device)
    labels = labels.to(device)
    # Forward Propagation
    logits = model(images)

    # Loss
    loss = criterion(logits, labels)
    train_loss += loss

    # accuracy
    acc = accuracy(logits, labels)
    train_acc += acc

    # zero_grad
    optimizer.zero_grad()
    # backward propagation
    loss.backward()
    # update gradient
    optimizer.step()
    
  # Average loss and accuracy
  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f'Trainning Loss: {train_loss.item():.3f} Training Accuracy: {train_acc.item():.3f}%')

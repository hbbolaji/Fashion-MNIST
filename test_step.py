import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def test_step(model:nn.Module, data_loader: DataLoader, criterion: nn.Module, accuracy, device: torch.device):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for (images, labels) in data_loader:
      images = images.to(device)
      labels = labels.to(device)
      # forward propagationn
      logits = model(images)

      # loss
      loss = criterion(logits, labels)
      test_loss += loss

      # accuracy
      acc = accuracy(logits, labels)
      test_acc += acc
    # Average test loss and accuracy
    test_acc /= len(data_loader)
    test_loss /= len(data_loader)
    print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f}')

    return test_acc, test_loss
  


import torch

def accuracy(logits:torch.Tensor, labels:torch.Tensor):
  prediction = torch.argmax(logits, dim=1)
  correct = (prediction == labels).sum()
  acc = 100 * correct / len(labels)
  return acc
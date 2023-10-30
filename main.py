import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from FashionMNISTModel import FashionMNISTModel
from train_step import train_step as ts
from helper_function import accuracy

# Hyperparameter
epochs = 5
lr = 0.1
batch_size = 32

# Device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transfrom
transform = transforms.Compose([transforms.ToTensor()])

# dataset
train_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model
torch.manual_seed(42)
model = FashionMNISTModel(input_shape= 28*28,
                          hidden_units=10,
                          output_shape=10).to(device)

# Optimizer and loss Function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training Loop
for epoch in range(1):
  ts(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, accuracy=accuracy, device=device)

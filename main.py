import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os.path
import matplotlib.pyplot as plt
from FashionMNISTModel import FashionMNISTModel
from FashionMNISTCNNModel import FashionMNISTCNNModel
from train_step import train_step
from test_step import test_step
from helper_function import accuracy, eval_model, visualize

# Hyperparameter
epochs = 10
lr = 0.01
batch_size = 32
exploration = False  # Change to True to see display 6 random train images and False to train the model

# Device agnostic
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transfrom
transform = transforms.Compose([transforms.ToTensor()])

# dataset
train_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

classes = train_set.classes
class_to_idx = train_set.class_to_idx

# dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size)

if exploration:
  samples = iter(train_loader)
  images, labels = next(samples)
  visualize(images, labels)
  plt.show()
else:
  # Model
  torch.manual_seed(42)
  model = FashionMNISTModel(input_shape= 28*28,
                            hidden_units=10,
                            output_shape=10).to(device)
  modelv1 = FashionMNISTCNNModel(1, 10, len(classes)).to(device)

  # Optimizer and loss Function
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(modelv1.parameters(), lr=lr)

  # Training and Testing Loop
  for epoch in range(epochs):
    print(f'{epoch + 1} / {epochs}')
    train_step(model=modelv1, data_loader=train_loader, criterion=criterion, optimizer=optimizer, accuracy=accuracy, device=device)
    test_step(modelv1, test_loader, criterion, accuracy, device)

  # Model Evaluation
  model_result = eval_model(modelv1, test_loader, criterion, accuracy)
  print(model_result)

# saving model
path = '' # Enter the path to save your model
if os.path.isfile(path):
  print('Model Already Saved')
else:
  torch.save(modelv1.state_dict(), './modelv1.pth')
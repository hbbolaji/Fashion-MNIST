import torch
from torchvision import transforms
import cv2 as cv
from FashionMNISTModel import FashionMNISTModel
from FashionMNISTCNNModel import FashionMNISTCNNModel
import numpy as np

# hyperparameters
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_to_idx = {'T-shirt/top': 0,
                'Trouser': 1,
                'Pullover': 2,
                'Dress': 3,
                'Coat': 4,
                'Sandal': 5,
                'Shirt': 6,
                'Sneaker': 7,
                'Bag': 8,
                'Ankle boot': 9}

# model = FashionMNISTModel(28*28, 10, 10)
# model.load_state_dict(torch.load('./model.pth'))

modelv1 = FashionMNISTCNNModel(1, 10, len(classes))
modelv1.load_state_dict(torch.load('./modelv1.pth'))

image = cv.imread('./images/image_f.jpg')
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
_, th = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY_INV)
kernel = np.ones((5, 5), np.uint8)
erode = cv.erode(th, kernel )
resized_image = cv.resize(erode, (28,28), interpolation=cv.INTER_CUBIC)

input_image = resized_image.reshape((28, 28, 1))
transformed = transforms.ToTensor()(input_image)
# print(transformed)

logits = modelv1(transformed.unsqueeze(dim=1))
prediction = torch.argmax(logits, dim=1).item()
predicted_class = classes[prediction]

while True:
  cv.imshow(f'Predicted class: {predicted_class}', image)
  key = cv.waitKey(1)
  if key == 27:
    break
cv.destroyAllWindows()
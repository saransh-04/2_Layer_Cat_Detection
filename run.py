import numpy as np
import matplotlib.pyplot as plt

from models.model import *

import h5py
import time
import scipy
from PIL import Image
from scipy import ndimage

np.random.seed(1)

### Inputing values
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# print(train_x_orig.shape, train_y.shape, test_x_orig.shape, test_y.shape)
# print(classes) ## --- Output - [b'non-cat' b'cat']


### Let's view one of the image from the dataset
index = 10  # considering the 10th image
plt.imshow(train_x_orig[index])  # This command will load the image
print("Y for the image = " + str(train_y[0, index]) + ". It is a - " +
      classes[train_y[0, index]].decode("utf-8") + "image.")
# Here the train_y[0, index] outputs whether it is a cat or not, as 0 0r 1.
# Then the classes[0 or 1] outputs b'non-cat', which we have to decode using the decode("utf-8").
plt.show()  # This will print the image with index 10

### Exploring our dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))
print("\n\n")

### Reshaping the training examples.
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

### Standardizing the data to keep the values in between 0 and 1
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("\n\n")

### Running the Model
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075

parameters, costs = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y), learning_rate=learning_rate,
                                    num_iterations=2500, print_cost=True)
plot_costs(costs, learning_rate)

### Predicting the Error
predictions_train = predict(train_x, train_y, parameters)
print("Training Accuracy: " + str(predictions_train))

predictions_test = predict(test_x, test_y, parameters)
print("Testing Accuracy: " + str(predictions_test))

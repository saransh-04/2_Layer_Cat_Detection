import numpy as np
import matplotlib.pyplot as plt
import h5py


# Function to Load Data
def load_data():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', 'r')  # used to read the file
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # training features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File('dataset/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # listing the classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # reshaping
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Initializing the Parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)  # To define a particular set of random variables.

    # defining the parameters
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Checking for the shape
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    # Defining the parameters
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


# Linear Forward Function


# Sigmoid Function
def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z

    assert (A.shape == Z.shape)  # To check if the shape and all are working fine.

    return A, cache


# ReLU Function
def relu(Z):
    A = np.maximum(0, Z)
    cache = Z

    assert (A.shape == Z.shape)  # To check if the shape and all are working fine.

    return A, cache


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


# Linear Activation Forward
def linear_activation_forward(A_prev, W, b, activation):
    global activation_cache, linear_cache

    A = []

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    # linear cache = (A, W, b)
    # activation cache = Z

    return A, cache


# Compute Cost
def compute_cost(AL, Y):
    m = AL.shape[1]  # number of examples

    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)), axis=1, keepdims=True) / m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


# Sigmoid Backwards
def sigmoid_backward(dA, cache):
    # cache = activation_cache = Z

    Z = cache

    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = np.multiply(dA, np.multiply(s, (1 - s)))

    assert (dZ.shape == Z.shape)

    return dZ


# ReLU Backwards
def relu_backward(dA, cache):
    # cache = activation_cache = Z

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


# Linear Backward
def linear_backward(dZ, cache):
    # cache = linear_cache = (A, W, b)

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev, dW, db


# Linear Activation Backward
def linear_activation_backward(dAL, cache, activation):
    global dA_prev, dW, db

    (linear_cache, activation_cache) = cache
    # linear cache = (A, W, b)
    # activation cache = Z

    if activation == "sigmoid":
        dZ = sigmoid_backward(dAL, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu_backward(dAL, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# Updating the Parameters
def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]

    return parameters


# Actual Model
def two_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):

    np.random.seed(1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize the Parameters:
    parameters = initialize_parameters(n_x, n_h, n_y)

    # W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Running Forward and Backward Propagation

    for i in range(0, num_iterations):

        # Forward Propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # Cost Computation
        cost = compute_cost(A2, Y)

        # Backward Propagation

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Computing other factors of Back Propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # The gradients
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        # Updating the Parameters

        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost after every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


# Plotting the Graph of cost and number of iterations
def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


# Predict Function
def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    A1, cache1 = linear_activation_forward(X, parameters["W1"], parameters["b1"], "relu")
    A2, cache2 = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

    for i in range(0, A2.shape[1]):
        if A2[0, i] < 0.5:
            p[0, i] = 0
        else:
            p[0, i] = 1

    accuracy = np.sum((p == Y) / m)

    return accuracy

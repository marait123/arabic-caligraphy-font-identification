import numpy as np

import torch
from torch import nn, optim

import torch
from torch import nn, optim

def nn_train(model, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs=100, lr = 0.1):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr)

    model.to(device)

    train_losses, validation_losses, test_losses = [], [], []
    train_losses = []
    for epoch in range(epochs):
        inputs, labels = X_train.to(device), Y_train.to(device)
        
        optimizer.zero_grad()

        log_ps = model(inputs.float())
        loss = criterion(log_ps, labels)

        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            train_loss = loss.item() / len(labels)
            train_losses.append(train_loss)
            
            with torch.no_grad():
                model.eval()
                inputs, labels = X_valid.to(device), Y_valid.to(device)
                log_ps = model(inputs.float())
                loss = criterion(log_ps, labels)
                valid_loss = loss.item()/len(labels)
                validation_losses.append(valid_loss)

                inputs, labels = X_test.to(device), Y_test.to(device)
                log_ps = model(inputs.float())
                loss = criterion(log_ps, labels)
                test_loss = loss.item()/len(labels)
                test_losses.append(test_loss)
            
            model.train()

            print(f'Epoch: {epoch+1}/{epochs}',
                f"Training Loss: {train_loss}",
                f"validation Loss: {valid_loss}",
                f"Test Loss: {test_loss}")

    return train_losses, validation_losses, test_losses

def nn_predict(model, features):
    with torch.no_grad():
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        features = features.to(device)
        logps = model(features.float())
        ps = torch.exp(logps.float())
        predictions = ps.argmax(dim=1)
    
    model.train()
    return predictions.to('cpu')

def nn_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    return correct/len(labels)
    

    


def linear_activation_forward(A_prev, W, b, activation):
        """
        Forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python tuple containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)    
        
        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = relu(Z)
            
        cache = (linear_cache, activation_cache)

        return A, cache

def linear_activation_backward(dA, cache, activation):
        """
        Backward propagation for the LINEAR->ACTIVATION layer.
        
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)        
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
        

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        
        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db


def sigmoid(Z):
    """
    Sigmoid activation function
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    RELU activation function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

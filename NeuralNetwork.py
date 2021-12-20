from nn_utils import *

"""
    use:
        layer_dims = [n_x, n_h1, n_h2, n_y]
        params, costs = NeuralNetwork().train(X, Y, layers_dims)
"""

class NeuralNetwork:
    def __init__(self):
        pass

    def initialize_parameters(self, layers_dims):
        """
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        parameters = {}
        L = len(layers_dims) # number of layers in the network

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

            assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))

        return parameters

    def forward_propagation(self, X, parameters):
        """
        forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- weights to be applied
        
        Returns:
        AL -- activation value from the output (last) layer
        caches
        """

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        # The for loop starts at 1 because layer 0 is the input
        for l in range(1, L):
            A_prev = A 
            A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)
            
        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
            
        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Cross entropy cost function

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())

        return cost

    def backward_propagation(self, AL, Y, caches):
        """
        Backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
        
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
        
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
        
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
        grads["dA" + str(L-1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads

    def update_parameters(self, params, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        params -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                    parameters["W" + str(l)] = ... 
                    parameters["b" + str(l)] = ...
        """

        parameters = params.copy()
        L = len(parameters) // 2 # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters

    def train(self, X, Y, layers_dims, learning_rate, num_iterations = 3000, print_cost=False):
        """
        Train the Neural network to learn parameters.
        
        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        parameters = self.initialize_parameters(layers_dims)

        costs = []                         # keep track of cost
        
        for i in range(0, num_iterations):
            AL, caches = self.forward_propagation(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            
            if print_cost and i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == num_iterations:
                costs.append(cost)
        
        return parameters, costs
    
    def predict(self, X, parameters):
        """
        Predict the results of the neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- learned parameters of the model

        Returns:
        p -- predictions for the given dataset X
        """
        
        # Forward propagation
        probas, _ = self.forward_propagation(X, parameters)

        # convert probas to 0/1 predictions
        p = np.where(probas > 0.5, 1, 0)

        #print results
        #print ("predictions: " + str(p))            
        return p
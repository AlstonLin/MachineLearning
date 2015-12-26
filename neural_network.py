import numpy as np
import math
import random

"""
DATA STRUCTURES
"""
class Node:
    def __init__(self, num_input):
        self._numInput = num_input
        self.error = 0 #For back-prop
        param_vector = []; #Param Vector
        #Initializes the weights / param vector to random vals near 0
        for i in range(num_input):
            param_vector.append(random.random())
        self._param_vector = np.array(param_vector)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + math.exp(-x))

    def get_output(self, vector):
        return Node.sigmoid(np.dot(self._param_vector, vector))

class Layer:
    def __init__(self, num_nodes, num_input):
        self._nodes = []
        self.num_nodes = num_nodes
        for i in range(num_nodes):
            self._nodes.append(Node(num_input + 1)); #Inputs + Bias

    def get_output(self, in_vector):
        out = []
        for i in range(self.num_nodes):
            out.append(self._nodes[i].get_output(in_vector))
        out_vector = np.array(out).transpose()
        return out_vector

    def set_error(self, param_index, val):
        self._nodes[param_index].error = val

    def back_prop(self, next_layer):
        """
        for i in range(next_layer.num_nodes):
            delta = next_layer.
            for j in range(self.num_nodes):
                self._nodes
        """
        pass

"""
NETWORK
"""
class XORNetwork:
    def __init__(self):
        #Input layer is not represented by a Layer Object; Just feed a Vector to the hidden layer
        self._hidden_layer = Layer(2, 2)
        self._output_layer = Layer(1, 2)

    def predict(self, in1, in2):
        in_vector = np.array([1, in1, in2]).transpose()
        hidden_vector = np.append(1, self._hidden_layer.get_output(in_vector))
        out_vector = self._output_layer.get_output(hidden_vector)
        return out_vector[0]

    def train(self, in1, in2, out):
        #Determines output
        predicted = self.predict(in1, in2)
        #Sets the error vals for the output layer
        self._output_layer.set_error(0, predicted - out)
        #Back props
        self._hidden_layer.back_prop(self._output_layer)

network = XORNetwork()
print(network.predict(0, 0))
print(network.predict(0, 1))
print(network.predict(1, 0))
print(network.predict(1, 1))

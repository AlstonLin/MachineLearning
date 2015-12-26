import numpy as np
import math
import random
"""
DATA STRUCTURES
"""
class Node:
    def __init__(self, numInput):
        self._numInput = numInput
        #Creates a random nxn matrix, where n = numInput
        vector = []; #Param Vector
        for i in range(numInput):
            vector.append(random.randint(1, 10))
        self._vector = np.array(vector)

    def getOutput(self, vector):
        product = 1.0 / (1 + math.exp(-np.dot(self._vector, vector)))
        return product

class Layer:
    def __init__(self, numNodes, numInput):
        self._nodes = []
        self._numNodes = numNodes
        for i in range(numNodes):
            self._nodes.append(Node(numInput + 1)); #Inputs + Bias

    def getOutput(self, inVector):
        out = []
        for i in range(self._numNodes):
            out.append(self._nodes[i].getOutput(inVector))
        outVector = np.array(out).transpose()
        return outVector

class XORNetwork:
    def __init__(self):
        #Input layer is not represented by a Layer Object; Just feed a Vector to the hidden layer
        self._hiddenLayer = Layer(2, 2)
        self._outputLayer = Layer(1, 2)

    def predict(self, in1, in2):
        inVector = np.array([in1, in2, 1]).transpose()
        hiddenVector = np.append(self._hiddenLayer.getOutput(inVector), 1)
        outVector = self._outputLayer.getOutput(hiddenVector)
        return outVector[0]

network = XORNetwork()
print(network.predict(0, 0))
print(network.predict(0, 1))
print(network.predict(1, 0))
print(network.predict(1, 1))

import numpy as np
import random
"""
DATA STRUCTURES
"""
class Node:
    def __init__(self, numInput):
        self._numInput = numInput
        #Creates a random nxn matrix, where n = numInput
        vector = [];
        for i in range(numInput):
            vector.append(random.randint(1, 10))
        self._vector = np.array(vector)

    def getOutput(self, vector):
        product = np.dot(self._vector, vector)
        return product

class Layer:
    def __init__(self, numNodes, numInput):
        self._nodes = []
        self._numNodes = numNodes
        for i in range(numNodes):
            self._nodes.append(Node(numInput));

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
        inVector = np.array([in1, in2]).transpose()
        hiddenVector = self._hiddenLayer.getOutput(inVector)
        outVector = self._outputLayer.getOutput(hiddenVector)
        return outVector[0]

network = XORNetwork()
print(network.predict(0, 1))

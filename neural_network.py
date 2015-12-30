import numpy as np
import math
import random

"""
DATA STRUCTURES
"""

def dot(vector1, vector2):
  product = 0
  for i in range(len(vector1)):
    product += vector1[i] * vector2[i]
  return product

class Node:
  LEARN_RATE = 1
  def __init__(self, num_input):
    self.numInput = num_input
    self._derivative_vector = [] #For gradient descent
    self.delta = 1 #For back-prop
    self._input_vector = [] #The previous layer's 'z' for back-prop
    param_vector = [] #Param Vector
    #Initializes the weights / param vector to random vals near 0
    for i in range(num_input):
      param_vector.append(random.random())
      self._param_vector = param_vector

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))
  
  def get_output(self, vector):
    self.output = Node.sigmoid(dot(self._param_vector, vector))
    self._input_vector = vector
    return self.output

  def get_weight(index):
    return self._param_vector[index]
    
  def calculate_derivative(self):
    self._derivative_vector = []
    for inp in self._input_vector
      self._derivative_vector.append(self.delta * inp) #dE/dw = delta * z

  def gradient_descent(self):
    for i in range(len(self._param_vector)):
      #Update Rule: weight = weight - a*(dE/dw)
      self._param_vector[i] = self._param_vector[i] - LEARN_RATE * self._derivative_vector[i]

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
    return out

  def calculate_delta(self, expected, actual):
    for i in range(self.num_nodes):
      node.delta = actual[i] - expected[i]
      node.calculate_derivative()
 
  def back_prop(self, next_layer):
    next_delta = next_layer.delta
    for i in range(self.num_nodes):
      activation_derivative = self._nodes[i].output * (1 - self._nodes[i].output) #h'(a) = z(1 - z)
      total = 0 #For sum(w*delta)
      for j in range(next_layer.num_nodes):
        total += next_layer._nodes[j].get_weight(i) * next_delta
        self._nodes[i].delta = activation_derivative * total #delta = h'(a) * sum(w*delta)
      self._nodes[i].calculate_derivative()

  def gradient_descent(self): 
    for node in self._nodes:
      node.gradient_descent()
  

"""
NETWORK
"""
class XORNetwork:
  def __init__(self):
    #Input layer is not represented by a Layer Object; Just feed a Vector to the hidden layer
    self._hidden_layer = Layer(2, 2)
    self._output_layer = Layer(1, 2)
 
  def predict(self, in1, in2):
    in_vector = [1, in1, in2]
    out =  self._hidden_layer.get_output(in_vector)
    hidden_vector = [1] + out
    out_vector = self._output_layer.get_output(hidden_vector)
    return out_vector[0]

  def train(self, in1, in2, out):
    #Determines output
    predicted = self.predict(in1, in2)
    error = predicted - out
    #Continue until practically converged
    while error > ERROR_TOLERANCE: 
      #Sets the delta vals for the output layer + derivative
      self._output_layer.calculate_delta(0, predicted - out)
      #Back prop
      self._hidden_layer.back_prop(self._output_layer)
      #Gradient Descent
      self._output_layer.gradient_descent()
      self._hidden_layer.gradient_descent()
      #Recalculates the error
      predicted = self.predict(in1, in2)
      error = predicted - out

network = XORNetwork()
print(network.predict(0, 0))
print(network.predict(0, 1))
print(network.predict(1, 0))
print(network.predict(1, 1))

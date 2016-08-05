import numpy as np

class LogisticRegression:
    def __init__(self, numParams, threshold, regularization=0):
        self._numParams = numParams
        self._theta = np.random.random(numParams)
        self._threshold = threshold
        self._lambda = regularization

    def validateXSize(self, X):
        if X.shape[1] != self._numParams:
            raise ValueError("The given X was a " + X.shape[0] + " * " + X.shape[1] + \
                    " matrix. Expected a " + m + " * " + self._numParams + "matrix.")
    
    def h(self, X):
        z = np.dot(X, self._theta)
        return 1 / (1 + np.exp(-z))
   
    """
    Only for testing purposes
    """
    def calculateGradNumeric(self, X, y, epsilon):
        grad = np.zeros(self._numParams)
        for i in range(self._numParams):
            clonedPos = np.copy(self._theta)
            clonedNeg = np.copy(self._theta)
            clonedPos[i] += epsilon
            clonedNeg[i] -= epsilon
            grad[i] = (self.calculateCost(X, y, theta=clonedPos) - self.calculateCost(X, y, theta=clonedNeg)) / (2 * epsilon)
        return grad

    def calculateAndCheckGradient(self, X, y):
        epsilon = 0.0001
        calculated = self.calculateGrad(X, y)
        numeric = self.calculateGradNumeric(X, y, epsilon)
        if np.abs(calculated - numeric).sum() > epsilon * self._numParams:
            print("Gradient Check failed! Numeric: ", numeric, ", Calculated: ", calculated)
            print("Additional Info - Theta: ", self._theta, ", Cost: ", self.calculateCost(X, y))
        return calculated

    """
    Calculates the cost using the current theta of this instance.
    
    Args:
        X - An m * n numpy matrix
        y - A numpy vector of size m
        theta - If given, will use this theta to calculate instead of self's
    Returns:
        A floating point representing the cost
    """
    def calculateCost(self, X, y, theta=None):
        self.validateXSize(X)
        m = X.shape[0]
        if theta is None: 
            h = self.h(X)
        else:
            z = np.dot(X, theta)
            h = 1 / (1 + np.exp(-z))
        J = -(y * np.log(h) + (1 - y) * np.log(1 - h)).sum() / m
        regularization = (self._lambda / (2 * m)) * (self._theta ** 2).sum()
        J = J + regularization
        return J

    """
    Calculates the the gradient vector of the Cost function wrt the change in theta (dJ / dTheta).

    Args:
        X - An m * n numpy matrix
        y - A numpy vector of size m
    Returns:
        A vector of size m representing the gradient vector
    """
    def calculateGrad(self, X, y):
        self.validateXSize(X)
        grad = np.zeros(self._numParams)
        m = X.shape[0]
        h = self.h(X)
        grad = np.dot(h - y, X) / m
        reg = (self._lambda / m) * self._theta
        return grad

    """
    Trains the algorithm by running gradient descent.

    Args:
        X - An m * n numpy matrix
        y - A numpy vector of size m of booleans
        alpha - The learning rate
        iter - The max number of iterations to run gradient descent
        checked - Use gradient checking to validate this works
    """
    def train(self, X, y, alpha=0.3, iter=10000, checked=False):
        self.validateXSize(X)
        for i in range(iter):
            grad = self.calculateAndCheckGradient(X, y) if checked else self.calculateGrad(X, y)
            self._theta = self._theta - np.multiply(grad, alpha)
    
    """
    Predicts a set of output values (y) for the given X.

    Args:
        X - An m * n numpy matrix
    Returns:
        A vector of size m representing all the predicted outputs
    """
    def predict(self, X):
        self.validateXSize(X)
        h = self.h(X)
        return h[:] > self._threshold

def runTest(X, y):
    reg = LogisticRegression(len(X[0]), 0.5, regularization=0.25);
    trainX = np.array(X)
    trainY = np.array(y)
    print("Cost before training: ", reg.calculateCost(trainX, trainY));
    reg.train(trainX, trainY, checked=True);
    print("Cost after training: ", reg.calculateCost(trainX, trainY));

def test():
    runTest([[-1, -1], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], [False, False, False, True, True, True])
    runTest([[0.5, 10.0], [0.75, 7.0], [1.00, 5.0], [1.25, 1.0], [1.50, 2.0], [2.0, 8.0], [3.0, 1.0], [2.5, 10.0], [2.0, 1.5]], [False, False, False, False, True, False, True, True, True])

test()

import numpy as np

class LinearRegression:
    def __init__(self, numParams):
        self._numParams = numParams
        self._theta = np.array([0] * numParams)

    def validateXSize(self, X):
        if X.shape[1] != self._numParams:
            raise ValueError("The given X was a " + X.shape[0] + " * " + X.shape[1] + \
                    " matrix. Expected a " + m + " * " + self._numParams + "matrix.")
    """
    Calculates the cost using the current theta of this instance.
    
    Args:
        X - An m * n numpy matrix
        y - A numpy vector of size m
    Returns:
        A floating point representing the cost
    """
    def calculateCost(self, X, y):
        self.validateXSize(X)
        m = X.shape[0]
        h = np.dot(X, self._theta)
        return ((h - y) ** 2).sum() / (2 * m)
    
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
        m = X.shape[0]
        h = np.dot(X, self._theta)
        grad = np.dot(h - y, X)
        return grad

    """
    Trains the algorithm by running gradient descent.

    Args:
        X - An m * n numpy matrix
        y - A numpy vector of size m
        alpha - The learning rate
        iter - The max number of iterations to run gradient descent
    """
    def train(self, X, y, alpha=0.05, iter=1000):
        self.validateXSize(X)
        for i in range(iter):
            grad = self.calculateGrad(X, y);            
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
        return np.dot(X, self._theta)       

def testRegression():
    reg = LinearRegression(2);
    trainX = np.array([[1, 1], [2, 2], [3, 3]])
    trainY = np.array([2.1, 3.9, 6.2]);
    print("Cost before training: ", reg.calculateCost(trainX, trainY));
    reg.train(trainX, trainY);
    print("Cost after training: ", reg.calculateCost(trainX, trainY));
    print("Predict [[4, 4], [-2, -2]]: ", reg.predict(np.array([[4, 4], [-2, -2]])))
testRegression()

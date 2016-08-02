"""
This algorithm implements linear regression using the Normal Equations. It also implements it's own linear algebra library (similar to numpy).
"""

import copy, random

"""
DATA STRUCTURES
"""

class Matrix:
    def __init__(self, numAttributes):
        assert numAttributes > 0, "Invalid number of attributes - " + numAttributes
        self._array = []
        self._numAttributes = numAttributes

    @staticmethod
    def createFromArray(array): #Should only be used to calculate determinants
        assert len(array) > 0, "Creating Matrix from empty array"
        numAttributes = len(array[0])
        matrix = Matrix(numAttributes)
        matrix._array = array;
        return matrix


    def addInput(self, params):
        assert len(params) == self._numAttributes, "Input has wrong number of attributes"
        #Inserts the vector into the array as a row
        self._array.append(params)

    def getArray(self):
        return self._array

    def getTranspose(self):
        if len(self._array) == 0:
            return []
        array = self.getArray()
        transpose = [[0 for x in range(len(self._array))] for x in range(len(self._array[0]))]
        for i in range(len(array)):
            for j in range(len(array[i])):
                transpose[j][i] = array[i][j]
        return Matrix.createFromArray(transpose)

    def getInverse(self): #Using the Adjoint method
        det = self.getDeterminant()
        assert det != 0, "Cannot invert a non-invertable matrix"
        adjunct = self.getCofactorMatrix().getTranspose()
        adjunct.scalarMultiply(1.0 / det)
        return adjunct


    def getCofactorMatrix(self):
        cof = [[0 for x in range(len(self._array[0]))] for x in range(len(self._array))]
        for i in range(len(self._array)):
            for j in range(len(self._array[i])):
                cof[i][j] = Matrix.getCofactor(self._array, i, j)
        return Matrix.createFromArray(cof)

    def getDeterminant(self):
        assert len(self._array) > 0, "Attempt to find determinant of an Empty Matrix"
        assert len(self._array) == len(self._array[0]), "Not an n x n Matrix: " + str(self._array)
        if len(self._array) == 1: #Base Case: 1x1 matrix
            det = self._array[0][0]
            return det
        if len(self._array) == 2: #Base Case: 2x2 matrix
            det = self._array[0][0] * self._array[1][1] - self._array[0][1] * self._array[1][0]
            return det

        det = 0
        i = 0; #Expand along the first row
        for j in range (len(self._array)):
            if self._array[i][j] != 0:
                cofactor = Matrix.getCofactor(self._array, i, j)
                det += self._array[i][j] * cofactor
        return det

    @staticmethod
    def getCofactor(array, i, j):
        assert len(array) > 0, "getMinor on an empty array"
        assert i < len(array) and i >= 0, "getMinor row is out of range"
        assert j < len(array[0]) and j >= 0, "getMinor column out of range"
        #Gets the minor matrix
        minor = copy.deepcopy(array)
        minor.pop(i) #Deletes row
        for row in minor: #Deletes column
            row.pop(j)
        minorMatrix = Matrix.createFromArray(minor)
        sign = (-1) ** (i + j)
        #Cofactor is the minor matrix's determinant
        cofactor = sign * minorMatrix.getDeterminant()
        return cofactor

    def scalarMultiply(self, multiple):
        for row in self._array:
            for i in range(len(row)):
                row[i] *= multiple

    @staticmethod
    def multiply(a, b):
        assert len(a._array) > 0 and len(b._array) > 0, "Cannot multiply empty matrices"
        assert len(a._array[0]) == len(b._array), "Cannot multiply matrices of different size: " + \
            str(a._array) + " and " + str(b._array)
        product = []
        rowsA = a.getRowVectors()
        columnsB = b.getColumnVectors()
        for row in rowsA:
            productRow = []
            for column in columnsB:
                dot = Vector.dotProduct(row, column)
                productRow.append(dot)
            product.append(productRow)
        return Matrix.createFromArray(product)

    def predict(self, params):
        return DataPoint([])

    def getArray(self):
        return copy.deepcopy(self._array)

    def getColumnVectors(self):
        columns = self.getColumns()
        columnVectors = []
        for column in columns:
            columnVectors.append(Vector(column))
        return columnVectors

    def getColumns(self):
        if len(self._array) == 0:
            return
        columns = [[] for x in range(len(self._array[0]))]

        for row in self._array:
            for i in range(len(self._array[0])):
                columns[i].append(row[i])
        return columns

    def getRowVectors(self):
        rowVectors = []
        for row in self._array:
            rowVectors.append(Vector(copy.deepcopy(row)))
        return rowVectors

class Vector:
    def __init__(self, array):
        self._array = array

    @staticmethod
    def dotProduct(a, b):
        assert len(a._array) == len(b._array), "Cannot dot vectors of different size: " + \
        str(a._array) + " * " + str(b._array)
        product = 0
        for i in range(len(a._array)):
            product += a._array[i] * b._array[i]
        return product

"""
MACHINE LEARNING ALGORITHM
"""
class LinearRegression:
    def __init__(self, numParams):
        self._matrix = Matrix(numParams + 1) #Accounting for the constant term
        self._targetValues = []
        self._paramsMatrix = None #Memoization

    def train(self, params, target):
        params.append(1.0) #For the Constant term
        self._matrix.addInput(params)
        self._targetValues.append(target)
        self._paramsMatrix = None

    def predict(self, params):
        params.append(1.0) #Constant term
        if self._paramsMatrix is None:
            self._paramsMatrix = self._matrix
            self._paramsMatrix = Matrix.multiply(self._paramsMatrix.getTranspose(), self._paramsMatrix)
            self._paramsMatrix = Matrix.multiply(self._paramsMatrix.getInverse(), self._matrix.getTranspose())
            self._paramsMatrix = Matrix.multiply(self._paramsMatrix, Matrix.createFromArray([self._targetValues]).getTranspose())
        paramsVector = Vector(self._paramsMatrix.getTranspose().getArray()[0])
        result = Vector.dotProduct(paramsVector, Vector(params))
        return result

"""
UNIT TESTING
"""
def assertAttribute(name, attributeName, expectedVal, actualVal):
    assert actualVal == expectedVal, "Wrong output with test " + name + " - " + \
        attributeName + " = " + str(actualVal) + " instead of " + str(expectedVal)

def testMatrix(name, array, determinant, transpose, cofactor, inverse, multiplyWith, product):
    matrix = Matrix.createFromArray(array)
    multiplyMatrix = Matrix.createFromArray(multiplyWith)

    determinantTest = matrix.getDeterminant()
    transposeTest = matrix.getTranspose().getArray()
    cofactorTest = matrix.getCofactorMatrix().getArray()
    inverseTest = matrix.getInverse().getArray()
    productTest = Matrix.multiply(matrix, multiplyMatrix).getArray()

    assertAttribute(name, "transpose", transpose, transposeTest)
    assertAttribute(name, "determinant", determinant, determinantTest)
    assertAttribute(name, "cofactor", cofactor, cofactorTest)
    assertAttribute(name, "inverse", inverse, inverseTest)
    assertAttribute(name, "product", product, productTest)

def unitTestMatrix(): #Unit tests the Matrix class
    tests = []
    #----DEFINES THE TEST CASES------
    #TEST CASE 1
    tests.append([
        #name
        "Test 1",
        #array
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],
        #determinant
        1,
        #transpose
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],
        #cofactor
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],
        #inverse
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],
        #multiplyWith
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]],
        #product
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
    ])

    #TEST CASE 2
    tests.append([
        #name
        "Test 2",
        #array
        [[3, 2, 3],
        [4, 5, 6],
        [7, 8, 10]],
        #determinant
        1,
        #transpose
        [[3, 4, 7],
        [2, 5, 8],
        [3, 6, 10]],
        #cofactor
        [[2, 2, -3],
        [4, 9, -10],
        [-3, -6, 7]],
        #inverse
        [[2, 4, -3],
        [2, 9, -6],
        [-3, -10, 7]],
        #multiplyWith,
        [[24, 24, 24],
        [3, 3, 3],
        [2, 4, 6]],
        #product
        [[84, 90, 96],
        [123, 135, 147],
        [212, 232, 252]]
    ])

    #------RUN TEST CASES----------
    for test in tests:
        testMatrix(test[0], test[1], test[2], test[3], test[4], test[5], test[6], test[7])


def testRegression(name, data, results, startPredict, errorTolerance): #startPredict - index to start predict instead of train
    #errorTolerance - set to -1 if None and print error
    regression = LinearRegression(len(data[0]))
    maxError = 0
    for i in range (startPredict):
        regression.train(data[i], results[i])
    for i in range (startPredict, len(data)):
        result = regression.predict(data[i])
        if results[i] != 0:
            error = abs((result - results[i]) / results[i])
        else:
            error = abs(result)
        if error > maxError:
            maxError = error
    if errorTolerance == -1:
        print name, "(", startPredict, " trained / ", len(data) - startPredict, " predict) -  maxError = ", round(maxError * 100, 5), "%"
    else:
        assert maxError < errorTolerance, "FAILED TEST CASE - " + name + " Max Error = " + str(maxError) + \
            " with Error Tolerance = " + str(errorTolerance)
    print "Predicted Function: ", regression._paramsMatrix.getArray()

def generateTestCase(numCases, numVars, lowest, largest, maxDeviation, func):
    data = []
    results = []
    for i in range (0, numCases):
        params = []
        #Generates input and output
        for j in range (0, numVars):
            params.append(random.randint(lowest, largest))
        results.append(func(params))
        #Adds randomness
        for j in range (0, numVars):
            params[j] += (2 * random.random() - 1) * maxDeviation
        data.append(params)
    return data, results

def unitTestRegresion(): #Will use last 2 to Prediction
    tests = []
    #TEST CASE 1 - f(x1, x2, x3, x4, x5) = 3(x1 + x2 + x3 + x4 + x5) - 3
    data1, results1 = generateTestCase(100, 5, -10, 10, 0.01, lambda params: 3 * (params[0] + params[1]
        + params[2] + params[3] + params[4]) - 3)
    tests.append([
        #name
        "TEST CASE f(x1,x2,x3,x4,x5) = 3(x1 + x2 + x3 + x4 + x5) - 3 : ",
        #data
        data1,
        #results
        results1,
        #startPredict
        85,
        #errorTolerance
        -1
    ])
    #TEST CASE 2 - f(x, y, z) = x + y + z - 1
    data2, results2 = generateTestCase(1000, 3, -15, 15, 0.001, lambda params: params[0] + params[1]
        + params[2] - 1)
    tests.append([
        #name
        "TEST CASE f(x,y,z) = x + y + z - 1 : ",
        #data
        data2,
        #results
        results2,
        #startPredict
        750,
        #errorTolerance
        -1
    ])
    #TEST CASE 3 - f(x1, x2, x3, x4, x5, x6. x7) = x1 - x2 + 5x3 - 1.5x4 - 99x5 + 0.1x6 -0.0019x7 - 6
    data2, results2 = generateTestCase(10000, 7, -50, 50, 0.001, lambda params: params[0] - params[1]
        + 5*params[2] - 1.5*params[3] - 99*params[4] + 0.1*params[5] - 0.0019* params[6] - 6)
    tests.append([
        #name
        "TEST CASE f(x1, x2, x3, x4, x5, x6. x7) = x1 - x2 + 5x3 - 1.5x4 - 99x5 + 0.1x6 - 0.0019x7 - 6 : ",
        #data
        data2,
        #results
        results2,
        #startPredict
        9000,
        #errorTolerance
        -1
    ])
    for test in tests:
        testRegression(test[0], test[1], test[2], test[3], test[4])

def unitTest():
    print "---------------UNIT TESTS------------------"
    unitTestMatrix()
    print "Passed all Matrix unit tests"
    unitTestRegresion()
    print "Passed all Regression unit tests"

"""
EXECUTABLE CODE
"""
unitTest()

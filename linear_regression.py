
import copy

"""
DATA STRUCTURES
"""

class Matrix:
    def __init__(self):
        self._values = []
        self._array = []

    def __init__(self, array): #Should only be used to calculate determinants
        self._array = array;
        self._values = []

    def addInput(input):
        assert len(self._values) == 0 or len(input) == len(self._values[0]), "Added input of different sizes"
        self._values.append(input)
        #Inserts the vector into the array
        self._array.append(input.getVector())

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
        return Matrix(transpose)

    def getInverse(self): #Using the Adjoint method
        det = self.getDeterminant()
        assert det != 0, "Finding the determinant of a non-invertible matrix!"
        adjunct = self.getCofactorMatrix().getTranspose()
        adjunct.scalarMultiply(1.0 / det)
        return adjunct

    def getCofactorMatrix(self):
        if len(self._array) == 0:
            return Matrix()
        cof = [[0 for x in range(len(self._array[0]))] for x in range(len(self._array))]
        for i in range(len(self._array)):
            for j in range(len(self._array[i])):
                cof[i][j] = Matrix.getCofactor(self._array, i, j)
        return Matrix(cof)

    def getDeterminant(self):
        assert len(self._array) > 0, "Attempt to find determinant of an Empty Matrix"
        assert len(self._array) == len(self._array[0]), "Not an n x n Matrix"

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
        minorMatrix = Matrix(minor)
        sign = (-1) ** (i + j)
        #Cofactor is the minor matrix's determinant
        cofactor = sign * minorMatrix.getDeterminant()
        return cofactor

    def scalarMultiply(self, multiple):
        for row in self._array:
            for i in range(len(row)):
                row[i] *= multiple

    def getDataPoints(self):
        return self._values

    def getArray(self):
        return self._array

class DataPoint:
    def __init__(self, map):
        self._map = map

    def getValue(self, name):
        return self._map[name]

    def getVector(self):
        return self._map.values()



def linearRegression():
    return

"""
UNIT TESTING
"""
def assertAttribute(name, attributeName, expectedVal, actualVal):
    assert actualVal == expectedVal, "Wrong output with test " + name + " - " + \
        attributeName + " = " + str(actualVal) + " instead of " + str(expectedVal)

def testMatrix(name, array, determinant, transpose, cofactor, inverse):
    matrix = Matrix(array)
    determinantTest = matrix.getDeterminant()
    transposeTest = matrix.getTranspose().getArray()
    cofactorTest = matrix.getCofactorMatrix().getArray()
    inverseTest = matrix.getInverse().getArray()

    assertAttribute(name, "transpose", transpose, transposeTest)
    assertAttribute(name, "determinant", determinant, determinantTest)
    assertAttribute(name, "cofactor", cofactor, cofactorTest)
    assertAttribute(name, "inverse", inverse, inverseTest)

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
        [-3, -10, 7]]
    ])

    #------RUN TEST CASES----------
    for test in tests:
        testMatrix(test[0], test[1], test[2], test[3], test[4], test[5])


"""
ACTUAL MACHINE LEARNING CODE
"""
#Start with the unit tests
unitTestMatrix()
print "Passed all Matrix unit tests"

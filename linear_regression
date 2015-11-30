
import copy

class Matrix:
    def __init__(self):
        self.values = []
        self.array = []

    def __init__(self, array): #Should only be used to calculate determinants
        self.array = array;
        self.values = []

    def addInput(input):
        assert self.values.size() == 0 or input.size() == self.values[0].size(), "Added input of different sizes"
        self.values.append(input)
        #Inserts the vector into the array
        self.array.append(input.getVector())

    def getArray(self):
        return self.array

    def transpose(self):
        if self.array.size() == 0:
            return []
        array = self.getArray()
        transpose = [[0] * self.array.size()] * self.array[0].size()
        for i in range(array.size()):
            for j in range(array[i].size()):
                transpose[j][i] = array[i][j]
        return transpose

    def invert(self):
        return

    def getCofactorMarrix(self):
        if self.array.size() == 0:
            return Matrix()
        cof = [[0] * self.array[0].size()] * self.array.size()
        for i in range(self.array.size()):
            for j in range(self.array[i].size()):
                cof[i][j] = getCofactor(self.array, i, j)
        return cof

    def determinant(self):
        assert self.array.size() > 0, "Attempt to find determinant of an Empty Matrix"
        assert self.array.size() == self.array[0].size(), "Not an n x n Matrix"
        if self.array.size() == 1:
            return self.array[0][0]

        det = 0
        i = 0; #Expand along the first row
        for j in range (self.array.size()):
            cofactor = getCofactor(self.array, i, j)
            det += (-1 * ((i + j) % 2)) * self.array[i][j] * cofactor
        return det

    def getCofactor(self, array, i, j):
        assert array.size() > 0, "getMinor on an empty array"
        assert i < array.size() and i >= 0, "getMinor row is out of range"
        assert j < array[0].size() and j >= 0, "getMinor column out of range"
        minor = copy.deepcopy(array)
        minor.pop(i)
        for row in range(minor.size()):
            row.pop(j)
        minorMatrix = Matrix(minor)
        cofactor = minorMatrix.determinant()
        return cofactor

class DataPoint:
    def __init__(self, map):
        self.map = map

    def getValue(self, name):
        return self.map[name]

    def getVector(self):
        return self.map.values()



def linearRegression():
    return

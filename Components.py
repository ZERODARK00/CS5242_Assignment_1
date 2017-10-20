"""
    Create basic components of neural network including full connected layer, activation layer and so on.
"""

import numpy as np
import copy
import csv
import os
import math

class Layer(object):
    """
        define parent class of layers
    """
    def __init__(self):
        pass

    def forward(self, inData):
        pass

    def backward(self, inGradient):
        pass

class FCLayer(Layer):
    """
        full connected layer including forward and backward method for training
    """
    def __init__(self, inNodesNum, outNodesNum):
        self.inNodesNum = inNodesNum
        self.outNodesNum = outNodesNum

        # normal initialization
        normalStd = 0.1
    
        # Xavier initialization
        xavierStd = math.sqrt(2.0/(self.inNodesNum+self.outNodesNum))
        
        # MSRA initialiazation
        msraStd = math.sqrt(2.0/self.inNodesNum)

        self.weights = np.random.normal(0, msraStd, (self.outNodesNum, self.inNodesNum)).astype(np.float)
        self.bias = np.zeros(self.outNodesNum).astype(np.float)


        # standard gradients
        self.weightsGrad = None
        self.biasGrad = None

        # true gradinets
        self.wGradient = np.zeros(self.weights.shape, dtype=np.float)
        self.bGradient = np.zeros(self.bias.shape, dtype=np.float)

        self.layerName = 'FCLayer'

        self.data = None

    def forward(self, inData):
        self.data = inData
        return np.add(np.dot(self.weights, self.data.T).T, self.bias)

    def backward(self, inGradient, lr=0.001): # batchSize = 1
        """
            inData is the results from last layer 
        """
        wGradient = np.dot(inGradient.T, self.data)
        bGradient = np.sum(inGradient, axis=0)
        outGradient = np.dot(inGradient, self.weights)

        self.weights = self.weights - lr * wGradient
        self.bias = self.bias - lr * bGradient
        self.wGradient = wGradient
        self.bGradient = bGradient

        #print "weight gradient ", wGradient
        #print "bias gradient ", bGradient

        return outGradient

class ReLULayer(Layer):
    """
        ReLU activation function
    """
    def __init__(self, rate=0):
        self.rate = rate
        self.layerName = 'ReLULayer'
        self. gradient = None
        self.data = None

    def forward(self, inData):
        self.data = inData
        return np.add(np.maximum(0, self.data), self.rate * np.minimum(0, self.data))

    def backward(self, inGradient):
        inShape = inGradient.shape
        gradient = inGradient.reshape(1,-1)
        data = self.data.reshape(1,-1)
        for i in range(data.shape[1]):
            if  data[0,i] >= 0 :
                continue
            else:
                gradient[0,i] = self.rate * gradient[0,i]
        self.gradient = gradient.reshape(inShape)
        return self.gradient

class CrossEntropy(object):
    """
        cross entropy cost function layer also including forward and backward method for training
    """
    def __init__(self):
        self.loss = 0
        self.prob = None
        self.data = None

    def forward(self, inData, inLabels):
        self.data = inData
        batchSize, outputNum = self.data.shape
        shift = np.max(self.data, axis=1).reshape(batchSize,-1).repeat([outputNum],1)
        shiftData = self.data - shift
        probDenominator = np.sum(np.exp(shiftData), axis=1)
        prob = []
        for i in range(len(probDenominator)):
            prob.append(np.exp(shiftData[i])/probDenominator[i])
        prob = np.array(prob)

        minval = 10**-10
        #print 'log prob is ', np.log(prob)
        self.loss = - np.sum(np.multiply(inLabels, np.log(prob.clip(minval)))) / batchSize
        self.prob = prob
        return self.loss, self.prob

    def backward(self, inLabels):
        batchSize = len(self.data)
        #print self.prob
        return (self.prob - inLabels).astype(np.float)

class ModelBuild(Layer):
    """
        to build the model needing to be trained
    """
    def __init__(self):
        self.layers = []
        self.num = 0

    def add(self, LayerList):
        self.layers.extend(LayerList)
        self.num = self.num + len(LayerList)
        return self

    def pop(self):
        self.layers.pop()
        return self
        
    def forward(self, inData):
        data = inData
        for layer in self.layers:
            data = layer.forward(data)
            #print 'Layer %s output is ' %layer.layerName, data
        return data

    def backward(self, inGradient, lr=0.001):
        gradient = inGradient
        for layer in self.layers[::-1]:
            if layer.layerName == 'FCLayer':
                gradient = layer.backward(gradient, lr=0.001)
            else:
                gradient = layer.backward(gradient)
            #print 'Layer %s gradient is ' %layer.layerName, gradient
        return gradient

    def loadWeights(self, fileName):
        outList = []
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                outList.append(row[1:])
        
        rowIndex = 0
        for layer in self.layers:
            if layer.layerName == 'FCLayer':
                rowNum = layer.inNodesNum
                weights = np.array(outList[rowIndex:rowIndex+rowNum], dtype=np.float)
                layer.weights = weights.T
                rowIndex = rowIndex + rowNum

    def loadBias(self, fileName):
        outList = []
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                outList.append(row[1:])

        rowIndex = 0
        for layer in self.layers:
            if layer.layerName == 'FCLayer':
                bias = np.array(outList[rowIndex], dtype=np.float)
                layer.bias = bias
                rowIndex = rowIndex + 1

    def loadWeightsGrad(self, fileName):
        outList = []
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                outList.append(row)
        rowIndex = 0
        for layer in self.layers:
            if layer.layerName == 'FCLayer':
                rowNum = layer.inNodesNum
                gradient = np.array(outList[rowIndex:rowIndex+rowNum], dtype=np.float)
                layer.weightsGrad = gradient.T
                rowIndex = rowIndex + rowNum
        #print   self.layers[-1].weightsGrad

    def loadBiasGrad(self, fileName):
        outList = []
        with open(fileName, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                outList.append(row)
        
        rowIndex = 0
        for layer in self.layers:
            if layer.layerName == 'FCLayer':
                gradient = np.array(outList[rowIndex], dtype=np.float)
                layer.biasGrad = gradient
                rowIndex = rowIndex + 1

    def saveWeightsGrad(self, fileName):
        if os.path.isfile(fileName):
            os.remove(fileName)
        with open(fileName, 'ab') as csvfile:
            for layer in self.layers:
                if layer.layerName == 'FCLayer':
                    np.savetxt(csvfile, layer.wGradient.T, delimiter=',')
        #print   self.layers[-1].weightsGrad

    def saveBiasGrad(self, fileName):
        if os.path.isfile(fileName):
            os.remove(fileName)
        with open(fileName, 'ab') as csvfile:
            for layer in self.layers:
                if layer.layerName == 'FCLayer':
                    np.savetxt(csvfile, layer.bGradient.reshape(1,-1), delimiter=',')
        
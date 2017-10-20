"""
    deal with input and output data, as well as provide data for training
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def Csv2Array(fileName, delimiter):
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        outList = []
        for row in reader:
            rowArray = np.array(row, dtype=np.float)
            outList.append(rowArray)
        return np.array(outList)

def Array2Csv(fileName, data, fmt = '%d,%.5f,%.2f'):
    np.savetxt(fileName, data, fmt)

def DrawCurve(x, y, style='r-'):
    plt.plot(x, y, style)
    plt.show()

def OneHot(value, classNum):
    outArray = np.zeros(classNum, dtype=np.float)
    outArray[int(value)] = 1
    return outArray


def LabelProvided(fileName, classNum):
    """
        trasform one single value of labels into one hot vector
    """
    outList = []
    labelArray = Csv2Array(fileName, ' ')
    rowsNum = labelArray.shape[0];
    for row in range(rowsNum):
        rowArray = OneHot(labelArray[row, 0], classNum)
        outList.append(rowArray)
    return np.array(outList)
    
def DataProvided(fileName, delimiter):
    """
        provide input data set
    """
    return Csv2Array(fileName, delimiter)

def TrainProvided(dataArray, labelArray, batchSize, epochNum):
    """
        provide input data and labels in batch size with random sequence
    """
    (dataSize, featureSize) = dataArray.shape
    seqList = []
    for i in range(epochNum+1): # make sure data size is big enough to be devided by batchSize
        seq = np.arange(dataSize)
        np.random.shuffle( seq )
        seqList.append(seq)
    seqArray = np.array(seqList).flatten()

    batchNum = epochNum * dataSize / batchSize + 1
    for i in range(batchNum):
        indexArray = seqArray[i*batchSize:(i+1)*batchSize]
        yield dataArray[indexArray, :], labelArray[indexArray, :]

'''
x = TrainProvided(DataProvided('../../Question2_123/x_test.csv',','), LabelProvided('../../Question2_123/y_test.csv',4),16,1)
print x.next()
'''
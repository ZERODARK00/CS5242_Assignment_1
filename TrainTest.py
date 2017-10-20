from Components import *
from DataProcessing import *

def ModelTrain(NodesList, trainInput, testInput, epochNum, name):
    # model build
    model = ModelBuild()
    NodesNum = len(NodesList)
    for i in range(NodesNum-1):
        model.add([FCLayer(NodesList[i], NodesList[i+1]), ReLULayer()])
    model.pop()

    # define loss function
    lossFunc = CrossEntropy()

    # define train data
    trainDataSet, trainLabelSet, trainBatchSize = trainInput
    trainData = TrainProvided(trainDataSet, trainLabelSet, trainBatchSize, epochNum)
    dataSize = len(trainDataSet)

    # times of training with batch
    batchNum = epochNum * len(trainDataSet) / trainBatchSize + 1

    # define test data
    testDataSet, testLabelSet = testInput
    testInterval = 10

    # learning rate
    lr = 0.01

    # train and test log (iteration, loss, accuracy)
    trainLog = []
    testLog =[]

    # training
    for i in range(batchNum):
        if i % 1000:
            lr = lr * 0.01

        (batchData, batchLabel) = trainData.next()
        trainLoss, trainAccuracy = ModelTest(model, lossFunc,[batchData, batchLabel])
        lossGradient = lossFunc.backward(batchLabel)
        model.backward(lossGradient, lr)

        fc1Sparsity = np.count_nonzero(model.layers[0].wGradient > 1e-6)/float(model.layers[0].wGradient.size)

        print name.upper() + " Trainning iteration %d, loss = %.5f, accuracy=%.2f%%" %(i, trainLoss, trainAccuracy)
        trainLog.append([i, trainLoss, trainAccuracy, fc1Sparsity])

        # one times of testing after testinterval times of training
        if i % testInterval == 0:
            testLoss, testAccuracy = ModelTest(model, lossFunc, testInput)
            print ' '*len(name)+" Testing loss = %.5f, accuracy=%.2f%%" %(testLoss, testAccuracy)
            #print "FC1 gradients ", model.layers[0].wGradient
            testLog.append([i/testInterval, testLoss, testAccuracy])


    # save loss log
    Array2Csv(name+'_train_log.csv', np.array(trainLog), '%d,%.5f,%.2f,%.2f')
    Array2Csv(name+'_test_log.csv', np.array(testLog))

    
    return model, lossFunc

def ModelTest(model,lossFunc, testInput):
    dataArray, labelArray = testInput
    outputs = model.forward(dataArray)
    loss, probs = lossFunc.forward(outputs, labelArray)
    accuracy = 0
    for i in range(len(dataArray)):
        maxIndex = probs[i,:].argmax()
        labelIndex = labelArray[i, :].argmax()
        if maxIndex == labelIndex:  
            accuracy = accuracy + 1
    accuracy = float(accuracy) / len(dataArray) * 100
    return loss, accuracy

# process train data and test data
batchSize = 64
epochNum = 50
trainDataArray = DataProvided('../../Question2_123/x_train.csv',',')
trainLabelArray = LabelProvided('../../Question2_123/y_train.csv', 4)
testDataArray = DataProvided('../../Question2_123/x_test.csv',',')
testLabelArray = LabelProvided('../../Question2_123/y_test.csv', 4)
trainInput = [trainDataArray, trainLabelArray, batchSize]
testInput = [testDataArray, testLabelArray]

# trainning

nodeList1 = [14, 100, 40, 4]
model1, lossFunc1 = ModelTrain(nodeList1, trainInput, testInput, epochNum, 'model1')

nodeList2 = [14]
for i in range(6):
    nodeList2.append(28)
nodeList2.append(4)
model2, lossFunc2 = ModelTrain(nodeList2, trainInput, testInput, epochNum, 'model2')

nodeList3 = [14]
for i in range(28):
    nodeList3.append(14)
nodeList3.append(4)
model3, lossFunc3 = ModelTrain(nodeList3, trainInput, testInput, epochNum, 'model3')

# testing

# print 'On test dataset: '
# loss, accuracy = ModelTest(model3, lossFunc3, testInput)
# print "Loss = %.5f, accuracy=%.2f%%" %(loss, accuracy)
# ModelTest(model2, lossFunc2, dataArray, labelArray)
# ModelTest(model3, lossFunc3, dataArray, labelArray)

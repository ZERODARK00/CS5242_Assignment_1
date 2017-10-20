from Components import *

def ModelGradSave(NodesList, ；):
    # model build
    model = ModelBuild()
    NodesNum = len(NodesList)
    for i in range(NodesNum-1):
        model.add([FCLayer(NodesList[i], NodesList[i+1]), ReLULayer()])
    model.pop()

    # load weights, bias and gradients
    model.loadWeights(fileList[0])
    model.loadBias(fileList[1])
    #print model.layers[-1].weightsGrad

    # define loss function
    lossFunc = CrossEntropy()

    # define input and label
    dataArray = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    labelArray = np.array([[0, 0, 0, 1]])

    # caculate gradients
    outLogits = model.forward(dataArray)
    loss, pred = lossFunc.forward(outLogits, labelArray)
    lossGradient = lossFunc.backward(labelArray)
    model.backward(lossGradient)

    # save gradients
    model.saveWeightsGrad(fileList[2])
    model.saveBiasGrad(fileList[3])

# define model parameters

nodeList1 = [14, 100, 40, 4]
fileList1 = ['../../Question2_4/c/w-100-40-4.csv',
             '../../Question2_4/c/b-100-40-4.csv',
             '../../Question2_4/c/dw-100-40-4.csv',
             '../../Question2_4/c/db-100-40-4.csv']

nodeList2 = [14]
for i in range(6):
    nodeList2.append(28)
nodeList2.append(4)
fileList2 = ['../../Question2_4/c/w-28-6-4.csv',
             '../../Question2_4/c/b-28-6-4.csv',
             '../../Question2_4/c/dw-28-6-4.csv',
             '../../Question2_4/c/db-28-6-4.csv']

nodeList3 = [14]
for i in range(28):
    nodeList3.append(14)
nodeList3.append(4)
fileList3 = ['../../Question2_4/c/w-14-28-4.csv',
             '../../Question2_4/c/b-14-28-4.csv',
             '../../Question2_4/c/dw-14-28-4.csv',
             '../../Question2_4/c/db-14-28-4.csv']

nodeLists = [nodeList1, nodeList2, nodeList3]
fileLists = [fileList1, fileList2, fileList3]
for i in range(3):
    print 'Save gradients of model %d: ' %(i+1)
    ModelGradSave(nodeLists[i], fileLists[i])
    print 'Done'

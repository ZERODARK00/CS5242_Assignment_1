from DataProcessing import *

model1TrainLog = Csv2Array('model1_train_log.csv', ',')
model1TestLog = Csv2Array('model1_test_log.csv', ',')

model2TrainLog = Csv2Array('model2_train_log.csv', ',')
model2TestLog = Csv2Array('model2_test_log.csv', ',')

model3TrainLog = Csv2Array('model3_train_log.csv', ',')
model3TestLog = Csv2Array('model3_test_log.csv', ',')

# Network No.3 training 
# plt.plot(model3TrainLog[:,0].astype('int'), model3TrainLog[:,-1])
# plt.title('MSRA')
# plt.xlabel('iteration')
# plt.ylabel('FC1 sparsity')
# plt.ylim([-0.5,1])

# sparsity comparison
f, axarr = plt.subplots(3, 1)
axarr[0].plot(model1TrainLog[:,0].astype('int'), model1TrainLog[:,-1])
axarr[0].set_title('Network No.1')
axarr[0].set_ylabel('FC1 sparsity')
axarr[0].set_ylim([-0.5, 1])

axarr[1].plot(model2TrainLog[:,0].astype('int'), model2TrainLog[:,-1])
axarr[1].set_title('Network No.2')
axarr[1].set_ylabel('FC1 sparsity')
axarr[0].set_ylim([-0.5, 1])

axarr[2].plot(model3TrainLog[:,0].astype('int'), model3TrainLog[:,-1])
axarr[2].set_title('Network No.3')
axarr[2].set_xlabel('iteration')
axarr[2].set_ylabel('FC1 sparsity')
axarr[0].set_ylim([-0.5, 1])

# training cost
# f, axarr = plt.subplots(3, 1)
# axarr[0].plot(model1TrainLog[:,0].astype('int'), model1TrainLog[:,1])
# axarr[0].set_title('Network No.1')
# axarr[0].set_ylabel('loss')
# axarr[0].set_ylim([0, 2])

# axarr[1].plot(model2TrainLog[:,0].astype('int'), model2TrainLog[:,1])
# axarr[1].set_title('Network No.2')
# axarr[1].set_ylabel('loss')
# axarr[0].set_ylim([0, 2])

# axarr[2].plot(model3TrainLog[:,0].astype('int'), model3TrainLog[:,1])
# axarr[2].set_title('Network No.3')
# axarr[2].set_xlabel('iteration')
# axarr[2].set_ylabel('loss')
# axarr[0].set_ylim([0, 2])

# test cost
# f, axarr = plt.subplots(3, 1)
# axarr[0].plot(model1TestLog[:,0].astype('int'), model1TestLog[:,1])
# axarr[0].set_title('Network No.1')
# axarr[0].set_ylabel('loss')
# axarr[0].set_ylim([0, 2])

# axarr[1].plot(model2TestLog[:,0].astype('int'), model2TestLog[:,1])
# axarr[1].set_title('Network No.2')
# axarr[1].set_ylabel('loss')
# axarr[2].set_ylim([0, 2])

# axarr[2].plot(model3TestLog[:,0].astype('int'), model3TestLog[:,1])
# axarr[2].set_title('Network No.3')
# axarr[2].set_xlabel('iteration')
# axarr[2].set_ylabel('loss')
# axarr[2].set_ylim([0, 2])

# accuracy
# train accuracy
# f, axarr = plt.subplots(3, 2)
# axarr[0,0].plot(model1TrainLog[:,0].astype('int'), model1TrainLog[:,2]/100.0)
# axarr[0,0].set_title('Training\nNetwork No.1')
# axarr[0,0].set_ylabel('accuracy')
# axarr[0,0].set_ylim([0, 1])

# axarr[1,0].plot(model2TrainLog[:,0].astype('int'), model2TrainLog[:,2]/100.0)
# axarr[1,0].set_title('Network No.2')
# axarr[1,0].set_ylabel('accuracy')
# axarr[1,0].set_ylim([0, 1])

# axarr[2,0].plot(model3TrainLog[:,0].astype('int'), model3TrainLog[:,2]/100.0)
# axarr[2,0].set_title('Network No.3')
# axarr[2,0].set_xlabel('iteration')
# axarr[2,0].set_ylabel('accuracy')
# axarr[2,0].set_ylim([0, 1])

# # test accuracy
# axarr[0,1].plot(model1TestLog[:,0].astype('int'), model1TestLog[:,2]/100.0)
# axarr[0,1].set_title('Testing\nNetwork No.1')
# axarr[0,1].set_ylabel('accuracy')
# axarr[0,1].set_ylim([0, 1])

# axarr[1,1].plot(model2TestLog[:,0].astype('int'), model2TestLog[:,2]/100.0)
# axarr[1,1].set_title('Network No.2')
# axarr[1,1].set_ylabel('accuracy')
# axarr[1,1].set_ylim([0, 1])

# axarr[2,1].plot(model3TestLog[:,0].astype('int'), model3TestLog[:,2]/100.0)
# axarr[2,1].set_title('Network No.3')
# axarr[2,1].set_xlabel('iteration')
# axarr[2,1].set_ylabel('accuracy')
# axarr[2,1].set_ylim([0, 1])

# axarr[1, 0].plot(trainLog[:,0].astype('int'), trainLog[:,2]*0.01)
# axarr[1, 0].set_title('training accuracy')
# axarr[1, 0].set_xlabel('iteration')
# axarr[1, 0].set_ylabel('accuracy')

# axarr[1, 1].plot(testLog[:,0].astype('int'), testLog[:,2]*0.01)
# axarr[1, 1].set_title('testing accuracy')
# axarr[1, 1].set_xlabel('iteration')
# axarr[1, 1].set_ylabel('accuracy')

plt.show()
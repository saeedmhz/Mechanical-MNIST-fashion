import fashion_MNIST_disponly as fashion
import sys
import os
import numpy as np

##########################################################################################
# input text files
##########################################################################################
class Data():
    def __init__(self, is_train=True):
        # set filename based on test/train
        self.fname = 'train' if is_train else 'test'
    def loadData(self):
        data = np.loadtxt('input_data/input_' + self.fname + '_fashion_MNIST_first100.txt')
        folder_name =  self.fname + '_data' # output folder  
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        data = data.reshape((-1,28,28))
        return self.fname, folder_name, data

    def processData(self, data):
        data_copy = np.zeros(data.shape)
        for j in range(len(data)):
            for k in range(len(data)):
                data_copy[j,k] = data[int(27-k),j] #jj is columns of input, kk is rows
        return data_copy

data_num = int(sys.argv[1])
is_train = bool(sys.argv[2])
fname, folder_name, dataset = Data(is_train).loadData()
Xdisp = np.array([[]]).reshape(0,28*28)
Ydisp = np.array([[]]).reshape(0,28*28)
for num in range(data_num):
    data = Data().processData(dataset[num])
    disp_x, disp_y = fashion.generate_dataset(data)
    Xdisp = np.vstack((Xdisp,disp_x))
    Ydisp = np.vstack((Ydisp,disp_y))

with open(folder_name + '/' + fname + '_dispx_14.npy', 'wb') as file:
    np.save(file, Xdisp)

with open(folder_name + '/' + fname + '_dispy_14.npy', 'wb') as file:
    np.save(file, Ydisp)

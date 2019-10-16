import numpy as np
import matplotlib.pyplot as plt
import os
'''
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import preprocessing
'''

#%matplotlib inline

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_from_file(file):
    dict = unpickle("D:\\Custom\\Semester 7\\MV\\Assignments\\2\\cifar-10-batches-py\\"+file)
    print("Unpacking {}".format(dict[b'batch_label']))
    X = np.asarray(dict[b'data'].T).astype("uint8")
    Yraw = np.asarray(dict[b'labels'])
    Y = np.zeros((10,10000))
    for i in range(10000):
        Y[Yraw[i],i] = 1
    names = np.asarray(dict[b'filenames'])
    return X,Y,names

def get_data():
    x_train = np.empty((3072,0)).astype("uint8")
    y_train = np.empty((10,0))
    n_train = np.empty(0)
    for b in range(1,6):
        fn = 'data_batch_' + str(b)
        X, Y, names = get_data_from_file(fn)
        x_train= np.append(x_train, X, axis=1)
        y_train= np.append(y_train, Y, axis=1)
        n_train= np.append(n_train, names)
    del X, Y
    
    fn = 'test_batch'
    x_test, y_test, n_test = get_data_from_file(fn)
    return x_train, y_train, n_train, x_test, y_test, n_test

def get_label_names(file):
    dict = unpickle("D:\\Custom\\Semester 7\\MV\\Assignments\\2\\cifar-10-batches-py\\"+file)
    L = np.asarray(dict[b'label_names'])
    return L

def visualize_image(X,Y,names, label_names, id):
    rgb = X[:,id]
    #print(rgb.shape)
    img = rgb.reshape(3,32,32).transpose([1, 2, 0])
    #print(img.shape)
    plt.imshow(img)
    plt.title("%s%s%s" % (names[id], ', Class = ',label_names[np.where(Y[:,id]==1.0)]) )
    plt.show()
    #dir = os.path.abspath("output/samples")
    #plt.savefig(dir+"/"+names[id].decode('ascii'))

'''
print("Starting ...")
L = get_label_names('batches.meta')
x_train, y_train, n_train, x_test, y_test, n_test = get_data()
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('n_train.shape = ', n_train.shape)
print('x_test.shape = ', x_test.shape)

print(y_train[:,43000])
visualize_image(x_train,y_train,n_train, L, 43000)
# print('X.shape = ', X.shape)
# print('Y.shape = ', Y.shape)
# visualize_image(X,Y,names,1000)
print('Done.')
'''
import numpy as np
import random
from read_cifar10_tx import get_label_names, get_data, visualize_image
# read_cifar10_tx is the code from Q2

def dense(ip, w, num_neu, back, de_dop, lr):
    '''
    fully connected layer
    ip      = flattened input
    w       = weight matrix (bias included)
    num_neu = no. of neurons in layer
    back    = set 0 if forward pass
    de_dop  = partial derivative of error wrt layer output for backprop
    lr      = learning rate

    returns
    op      = layer output
    w       = weight matrix (bias included)
    de_dw   = partial derivative of error wrt weights and biases
    '''

    ip = np.append(ip, [1])         # to account for bias
    num_ip = len(ip)

    if w.shape != (num_neu, num_ip):
        w = np.random.randn(num_neu, num_ip)*0.01    ###
    
    if back == 0:
        op = np.matmul(w, ip)
        return op, w
    else:
        # backprop
        de_dop = np.tile(de_dop, (num_ip,1))

        de_dop = de_dop.transpose()

        de_dw = np.empty((num_neu, num_ip))
        for row_num in range(len(de_dw)):
            de_dw[row_num] = np.multiply(de_dop[row_num],ip)
        
        # w = w - lr*de_dw
        return 1, de_dw

def relu_f(ip):
    '''
    relu
    ip  = array

    returns
    y   = array
    '''

    y = ip
    for i in range(len(ip)):
        y[i] = max(0, ip[i])
    return y

def relu_diff(ip):
    '''
    derivative of relu
    ip  = array

    returns
    y   = array
    '''

    y = ip
    for i in range(len(ip)):
        if ip[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    
    return y

def relu_act(ip, back, w2, de_dop2):
    '''relu activation
    ip      = array
    back    = set 0 if forward pass
    w2      = weight matrix (bias included) of next layer
    de_dop2 = partial derivative of error wrt next layer output for backprop

    returns
    in forward pass     = array
    in backward pass    = partial derivative of error wrt input for backprop
    '''

    if back == 0:
        return relu_f(ip)
    else:
        #backprop
        num_ip = len(ip)
        tmp1 = np.matmul(w2.transpose(), de_dop2)
        tmp2 = relu_diff(ip)
        de_din = np.multiply(tmp1[:num_ip], tmp2)
        return de_din

def softmax_1d(X):
    '''
    softmax for array
    '''
    exps = np.exp(X)
    return exps / np.sum(exps)

def stable_softmax_1d(X):
    '''
    stable softmax for array
    '''
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def stable_softmax_2d(ip):
    '''
    stable softmax function for matrix
    '''
    exp_x = np.exp(ip - np.max(ip, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_diff(ip):
    '''
    derivative of softmax function
    '''

    p = stable_softmax_2d(ip)
    return np.multiply(p, 1-p)

def cross_entropy_loss(X, y):
    """
    X = output from fully connected layer (num_examples x num_classes)
    y = targetss (num_examples x 1)
    y is not one-hot encoded. 
    y can be computed as y.argmax(axis=1) from one-hot encoded vectors if required.
    """
    
    m = y.shape[0]
    p = stable_softmax_2d(X)
    
    #print(X[range(m), y])
    #print(p[range(m), y])
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    
    y = np.eye(10)[y]   #one hot
    de_dx = p - y
    
    '''
    dx = p.copy()
    dx[range(m), y] -= 1
    dx /= m                     ###
    de_dx = dx
    '''
    return loss, de_dx

def batch(x, y, size):
    '''
    make batches ignoring last few samples
    x       = input array   (num samples x sample data)
    y       = target array  (num samples x 1)
    size    = batch size

    returns
    x_batch = batched input     (num batches x batch size x sample data)
    y_batch = batched target    (num batches x batch size x 1)
    '''
    
    x_batch = []
    x_mini = []
    y_batch = []
    y_mini = []
    for i in range(len(x)):
        x_mini.append(x[i])
        y_mini.append(y[i])
        if len(x_mini) == size:
            
            x_batch.append(x_mini)
            y_batch.append(y_mini)
            x_mini = []
            y_mini = []
        
    return x_batch, y_batch

def shuffle(x_batch, y_batch):
    '''
    shuffle the batches
    '''
    tmp = zip(x_batch, y_batch)
    tmp = list(tmp)
    random.shuffle(tmp)
    x_b, y_b = zip(*tmp)
    return x_b, y_b

def train(x_train, y_train, batch_size, lr, epochs, x_test, y_test):
    '''
    batch train with SGD
    x_train     = training input array   (num samples x sample data)
    y_train     = training target array  (num samples x 1)
    batch_size  = batch size
    lr          = learning rate
    epochs      = epochs
    x_test      = testing input array   (num samples x sample data)
    y_test      = testing target array  (num samples x 1)

    returns
    w1          = weight matrix (bias included) of dense layer 1
    w2          = weight matrix (bias included) of dense layer 2
    '''

    x_batch, y_batch = batch(x_train, y_train, batch_size)
    #w1 = np.empty((0,0))
    #w2 = np.empty((0,0))
    w1 = np.random.randn(512, 3073)*0.01         ###
    w2 = np.random.randn(10, 513)*0.01           ###

    for i in range(epochs):
        x_batch, y_batch = shuffle(x_batch, y_batch)

        for j in range(len(x_batch)):
            y_mini = np.asarray(y_batch[j])
            op1_batch = []
            relu_out_batch = []
            op2_batch = []

            for k in range(batch_size):
                ip = x_batch[j][k]
                ip = ip/255
                
                op1, w1 = dense(ip, w1, 512, 0, 0, 0)
                op1_batch.append(op1)
                relu_out = relu_act(op1, 0, 0, 0)
                relu_out_batch.append(relu_out)
                op2, w2 = dense(relu_out, w2, 10, 0, 0, 0)
                op2_batch.append(op2)
            
            loss, de_dop2 = cross_entropy_loss(np.asarray(op2_batch),y_mini)
            #print('Batch Loss = ' + str(loss))
            
            de_dw1_tot = 0
            de_dw2_tot = 0
            de_dw1_batch = []
            de_dw2_batch = []
            de_dop1_batch = []

            for m in range(batch_size):
                ip = x_batch[j][m]
                ip = ip/255

                chk2, de_dw2 = dense(relu_out_batch[m], w2, 10, 1, de_dop2[m], 0)
                de_dw2_tot += de_dw2
                de_dw2_batch.append(de_dw2)
                de_dop1 = relu_act(op1_batch[m], 1, w2, de_dop2[m])
                de_dop1_batch.append(de_dop1)
                chk1, de_dw1 = dense(ip, w1, 512, 1, de_dop1, 0)
                de_dw1_tot += de_dw1
                de_dw1_batch.append(de_dw1)
            
            w1 = w1 - de_dw1_tot*lr/batch_size         ###
            w2 = w2 - de_dw2_tot*lr/batch_size         ###
        
        print(i+1)
        train_acc = accuracy(x_train, y_train, w1, w2)
        test_acc = accuracy(x_test, y_test, w1, w2)

        print("Train Accuracy: {}%".format(train_acc * 100 ))
        print("Test Accuracy: {}%".format(test_acc * 100 ))
    return w1, w2

def get_cifar10(num_train, num_test):
    '''
    read cifar10
    num_train       = no of training samples
    num_test        = no of test samples

    returns
    x_train     = training input array   (num_train x sample data)
    Y_train     = training target array  (num_train x 1)
    x_test      = testing input array   (num_test x sample data)
    Y_test      = testing target array  (num_test x 1)
    '''
    N = num_train
    M = num_test

    ## read dataset
    #print("Reading dataset")
    L = get_label_names('batches.meta')
    x_train, y_train, n_train, x_test, y_test, n_test = get_data()

    x_train = x_train[:,:N]
    x_train = x_train.transpose((1,0))
    y_train = y_train[:,:N]

    Y_train = []
    for id in range(N):
        i = np.where(y_train[:,id]==1.0)
        Y_train.append(i[0][0])
    Y_train = np.asarray(Y_train)

    #y_train = y_train.transpose([1,0])
    n_train = n_train[:N]

    x_test = x_test[:,:M]
    x_test = x_test.transpose((1,0))
    y_test = y_test[:,:M]

    Y_test = []
    for id in range(M):
        i = np.where(y_test[:,id]==1.0)
        Y_test.append(i[0][0])
    Y_test = np.asarray(Y_test)

    # y_test = y_test.transpose([1,0])
    n_test = n_test[:M]

    return x_train, Y_train, x_test, Y_test

def accuracy(x, y, w1, w2):
    '''
    calculate accuracy
    x       = input array   (num samples x sample data)
    y       = target array  (num samples x 1)
    w1      = calculated weight matrix (bias included) of dense layer 1
    w2      = calculated weight matrix (bias included) of dense layer 2
    '''

    correct = 0

    for i in range(len(x)):
        ip = x[i]/255
        t = y[i]

        op1, w1 = dense(ip, w1, 512, 0, 0, 0)
        #op1_batch.append(op1)
        relu_out = relu_act(op1, 0, 0, 0)
        #relu_out_batch.append(relu_out)
        op2, w2 = dense(relu_out, w2, 10, 0, 0, 0)
        #op2_batch.append(op2)
        out = stable_softmax_1d(op2)
        choice = np.argmax(out)

        if choice == t:
            correct += 1

    return correct/len(x)

x_train, Y_train, x_test, Y_test = get_cifar10(1000, 250)
#print(x_train[1][:10])
W1, W2 = train(x_train, Y_train, 32, 0.01, 10, x_test, Y_test)

'''
train_acc = accuracy(x_train, Y_train, W1, W2)
test_acc = accuracy(x_test, Y_test, W1, W2)

print("Train Accuracy: {}%".format(train_acc * 100 ))
print("Test Accuracy: {}%".format(test_acc * 100 ))
'''
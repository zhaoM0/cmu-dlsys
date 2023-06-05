import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    return x + y


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # parse image file
    with gzip.open(image_filename, 'rb') as f:
        image_cont = f.read()
    
    offset = 0
    fmt_hearder = '>iiii'
    magic, nums, rows, cols = struct.unpack_from('>iiii', image_cont, offset)
    # print("{}, {}, {}, {}".format(magic, nums, rows, cols))

    offset += struct.calcsize(fmt_hearder)
    fmt_image = '>' + str(rows * cols) + 'B'

    images = np.zeros((nums, rows * cols), dtype=np.float32)    
    for i in range(nums):
        im = struct.unpack_from(fmt_image, image_cont, offset)
        images[i] = np.array(im).astype(np.float32) / 255.
        offset += struct.calcsize(fmt_image)
    
    # parse label file
    with gzip.open(label_filename, 'rb') as f:
        label_cont = f.read()

    offset = 0
    fmt_hearder = '>ii'
    magic, nums = struct.unpack_from(fmt_hearder, label_cont, offset)
    # print("{}, {}".format(magic, nums))

    offset += struct.calcsize(fmt_hearder)
    labels = []
    fmt_label = '>B'

    for i in range(nums):
        lb = struct.unpack_from(fmt_label, label_cont, offset)
        labels.extend(lb);
        offset += struct.calcsize(fmt_label)

    labels = np.array(labels).astype(np.uint8)

    return images, labels


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    assert(Z.shape[0] == y.size)
    exp_Z = np.exp(Z)
    sum_exp_Z = np.sum(exp_Z, axis=1)       # (batch_size, )
    lce = np.log(sum_exp_Z) - Z[np.arange(0, y.size), y]
    lce = np.mean(lce)
    return lce


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    # tranform y to one hot matrix with shape(num_smaples x k) 
    class_num = theta.shape[1]
    samples_num = y.size
    y_one_hot = np.identity(class_num)[y]

    # iteration
    for start in range(0, samples_num, batch):
        batch_X = X[start : start + batch, :]
        batch_y = y_one_hot[start : start + batch, :]

        exp_XT  = np.exp(np.matmul(batch_X, theta))
        batch_Z = exp_XT / np.sum(exp_XT, axis=1, keepdims=True)

        # calculate gradient 
        batch_grad = np.matmul(batch_X.T, batch_Z - batch_y) / batch
        # SGD step
        theta[:] = theta - lr * batch_grad

# helper function         
def relu(out):
    return np.maximum(0, out)

def grad_relu(out):
    return np.where(out > 0, 1, 0)


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    class_num  = W2.shape[1];
    sample_num = y.size
    y_one_hot  = np.identity(class_num)[y]
    
    for start in range(0, sample_num, batch):
        batch_X = X[start : start + batch, :]
        batch_y = y_one_hot[start : start + batch, :]
        
        # forward process 
        Z2 = relu(np.matmul(batch_X, W1))
        Z3 = np.matmul(Z2, W2)
        
        exp_Z3 = np.exp(Z3)
        norm_Z3 = exp_Z3 / np.sum(exp_Z3, axis=1, keepdims=True)
        
        # backward process
        grad_W2 = Z2.T @ (norm_Z3 - batch_y)
        grad_W1 = batch_X.T @ ((norm_Z3 - batch_y) @ W2.T * grad_relu(batch_X @ W1))

        # SGD process
        W2[:] = W2 - lr * (grad_W2 / batch)
        W1[:] = W1 - lr * (grad_W1 / batch)


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

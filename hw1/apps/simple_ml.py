import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # parse image file
    with gzip.open(image_filesname, 'rb') as f:
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

def softmax_loss_numpy(Z, y_one_hot):
    exp_z = np.exp(Z)
    sum_expz = np.sum(exp_z, axis=1)
    lce = np.log(sum_expz) - np.sum(Z * y_one_hot, axis=1)
    return np.mean(lce)

def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_num = Z.shape[0]
    exp_z = ndl.ops.exp(Z)
    sum_exp_z = ndl.ops.summation(exp_z, axes = (1,))
    lce = ndl.ops.log(sum_exp_z) - ndl.ops.summation(ndl.ops.multiply(Z, y_one_hot), axes = (1,))
    sum_lce = ndl.ops.summation(lce)
    return ndl.ops.divide_scalar(sum_lce, batch_num)


def nn_epoch(X, y, W1, W2, lr = 0.1, batch = 100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    # raise NotImplementedError()
    samples_nums = X.shape[0]
    classes_nums = W2.shape[1]
    
    y_one_hot = np.identity(classes_nums)[y]    # shape (num_examples, num_classes)
    
    for start in range(0, samples_nums, batch):
        batch_X = ndl.Tensor(X[start : start + batch, :])
        batch_y = ndl.Tensor(y_one_hot[start : start + batch, :])
        
        # forward 
        h1 = ndl.ops.relu(ndl.ops.matmul(batch_X, W1))
        h2 = ndl.ops.matmul(h1, W2)
        
        loss = softmax_loss(h2, batch_y)
        loss.backward()
        
        W1 = (W1 - lr * W1.grad).detach()
        W2 = (W2 - lr * W2.grad).detach()

    return (W1, W2)
    

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)


if __name__ == "__main__":
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
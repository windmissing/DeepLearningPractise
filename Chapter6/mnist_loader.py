#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

#### Load the MNIST data
def get_part(data, seed, size):
    X, y = data
    n = X.shape[0]
    if n <= size:return data
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(n)
    part_indexes = shuffle_indexes[:size]
    return X[part_indexes], y[part_indexes]
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros(10)
    e[j] = 1.0
    return e

def vectorize(data):
    vector_y = [vectorized_result(y) for y in data[1]]
    return (data[0], np.array(vector_y))
    #return (data[0], data[1])

def load_data_shared(filename, seed, train_size, vali_size, test_size):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    
    return get_part(training_data, seed, train_size),\
           get_part(validation_data, seed, vali_size),\
           get_part(test_data, seed, test_size)
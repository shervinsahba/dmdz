import numpy as np

def print_array(x, precision=6, suppress=True):
    with np.printoptions(precision=precision, suppress=suppress):
        print(x)
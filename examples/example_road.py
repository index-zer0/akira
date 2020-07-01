from ctypes import *
import os

akira_file = os.getcwd() + "/akira.so"
akira = CDLL(akira_file)
print(type(akira))
size = (c_int * 2)(2, 1)  # create an array of two ints
network = akira.nn_constructor(1, size)
training_input = akira.matrix_constructor(2, 1);
training_output = akira.matrix_constructor(1, 1);

for i in range(10):
    #akira.train(network, training_input, training_output)
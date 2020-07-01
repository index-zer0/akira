from ctypes import *
import os

akira_file = os.getcwd() + "/akira.so"
print (akira_file)
akira = CDLL(akira_file)
print(type(akira))
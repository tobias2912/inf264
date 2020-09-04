import numpy as np
import csv
from tree import * 

def get_matrix(liste):
    return liste[:, :4]

def get_X(matrix):
    return matrix[:, :3]

def get_lable(liste):
    return liste[:,4]

def open_file(fileName): 
    return np.loadtxt(open(fileName, "rb"), delimiter=",")




if __name__ == "__main__":
    liste = open_file('data_banknote_authentication.txt')
    X = get_matrix(liste)
    y = get_lable(liste)
    print(X)
    tree = Decision_tree()
    tree.learn(X, y, tree.root)


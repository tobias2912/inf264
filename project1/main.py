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

def split(matrix):



if __name__ == "__main__":
    liste = open_file('data_banknote_authentication.txt')
    print('LISTE: ', liste)
    print('X: ', get_matrix(liste))
    print('LABEL: ', get_lable(liste))
    X = get_X(liste)
    y = get_lable(liste)
    tree = Decision_tree(X, y)


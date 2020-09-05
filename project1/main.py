import numpy as np
import csv
from tree import * 

def get_X(liste):
    return liste[:, :4]


def get_lable(liste):
    return liste[:,4]

def open_file(fileName): 
    return np.loadtxt(open(fileName, "rb"), delimiter=",")

if __name__ == "__main__":
    liste = open_file('project1/data_banknote_authentication.txt')
    X = get_X(liste)
    y = get_lable(liste)
    print(X)
    tree = Decision_tree()
    tree.learn(X, y, tree.root)
    print(tree)


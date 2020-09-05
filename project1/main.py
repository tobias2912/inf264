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
    tree.print_tree()
    new = np.array([3.6216,8.6661,-2.8073,-0.44699])
    tree.predict(new, tree.root)


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
    liste = open_file('data_banknote_authentication.txt')
    X = get_X(liste)
    y = get_lable(liste)
    X_train, X_test = X[:1000], X[1000:]
    y_train, y_test = y[:1000], y[1000:]
    print(f"train: {X_train}")
    print(f"test:{X_test}")
    tree = Decision_tree()
    tree.learn(X_train, y_train, tree.root)
    
    #tree.print_tree()
    new = np.array([-0.10234,1.8189,-2.2169,-0.56725])
    val = tree.predict(tree.root, new)
    print(f"value is {val}")
    wrong = 0
    correct = 0
    for x in X_test
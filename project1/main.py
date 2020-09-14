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
    liste = open_file('test_data.txt')
    X = get_X(liste)
    y = get_lable(liste)
    X_train, X_test = X[:1000], X[1000:]
    y_train, y_test = y[:1000], y[1000:]
    tree = Decision_tree()
    tree.learn(X_train, y_train, tree.root)
    wrong = 0
    correct = 0
    for rownumber, x in enumerate(X_train):
        predicted_val = tree.predict(tree.root, x)
        if predicted_val == y_train[rownumber]:
            correct += 1
        else:
            wrong += 1
    print(correct, wrong)
    print(f"correctness {correct/(correct+wrong)}")
    print("testing data:")
    wrong = 0
    correct = 0
    for rownumber, x in enumerate(X_test):
        predicted_val = tree.predict(tree.root, x)
        if predicted_val == y_train[rownumber]:
            correct += 1
        else:
            wrong += 1
    print(correct, wrong)
    print(f"correctness {correct/(correct+wrong)}")
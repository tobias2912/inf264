import numpy as np
import csv
from tree import * 
import os
import sys
import random
import time

def get_X(liste):
    return liste[:, :4]

def get_label(liste):
    return liste[:,4]

def open_file(fileName): 
    return np.loadtxt(open(os.path.join(sys.path[0], fileName), "r"), delimiter=",")

def split(matrix, percentage):
    '''splits a matrix row-wise with a given percentage and a set random seed'''
    test = []
    training = []
    random.seed(10)

    for row in matrix:
        if random.random() < percentage:
            test.append(row)
        else:
            training.append(row)
    return np.array(test), np.array(training)


if __name__ == "__main__":
    matrix = open_file('data_banknote_authentication.txt')
    X = get_X(matrix)
    y = get_label(matrix)

    start_time = time.time()

    test, train_pruning = split(matrix, 0.3)
    pruning_data, train = split(train_pruning, 0.3)
    X_test, X_train, X_pruning = get_X(test), get_X(train), get_X(pruning_data)
    y_test, y_train ,y_pruning = get_label(test), get_label(train), get_label(pruning_data)

    tree = Decision_tree()
    tree.learn(X_train, y_train, tree.root, X_pruning, y_pruning, prune=True, impurity_measure="gini")
    '''testing on training data'''
    wrong = 0
    correct = 0
    for rownumber, x in enumerate(X_train):
        predicted_val = tree.predict(tree.root, x)
        if predicted_val == y_train[rownumber]:
            correct += 1
        else:
            wrong += 1
    print(f"correct predictions: {correct}, wrong predictions: {wrong}")
    print(f"accuracy {correct/(correct+wrong)}")

    '''testing on test data'''
    print("------\ntesting data:")
    wrong = 0
    correct = 0
    for rownumber, x in enumerate(X_test):
        predicted_val = tree.predict(tree.root, x)
        if predicted_val == y_test[rownumber]:
            correct += 1
        else:
            wrong += 1
    print(f"correct predictions: {correct}, wrong predictions: {wrong}")
    print(f"accuracy {correct/(correct+wrong)}")
    print("--- %s seconds ---" % (time.time() - start_time))

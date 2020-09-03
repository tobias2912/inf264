import numpy as np
import csv

def get_X(liste):
    return liste[:, :4]

def get_lable(liste):
    return liste[:,4]

def open_file(fileName): 
    return np.loadtxt(open(fileName, "rb"), delimiter=",")

if __name__ == "__main__":
    liste = open_file('data_banknote_authentication.txt')
    print('LISTE: ', liste)
    print('X: ', get_X(liste))
    print('LABEL: ', get_lable(liste))


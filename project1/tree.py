from numpy import *
class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data


    def PrintTree(self):
        print(self.data)

    def insert(self, data):
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

class decision_tree:

    def __init__(self):
        root = Node("")


    def learn(X, y, impurity_measure):
        pass

    def predict(x):
        pass
    def IG(x): return H(y) - H2(y, x)

    def H(x) =  

    def P(x_val, X):


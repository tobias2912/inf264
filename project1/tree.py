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

class Decision_tree:

    def __init__(self):
        root = Node("")

    def learn(X, y, impurity_measure='entropy'):
        pass

    def predict(x):
        pass
    
    def less(x, y):
        return x<y
    def greater(x, y):
        return x>y

    #information gain
    def IG(x, col, split, X): 
        ig = H(y) - Hcond(x, split, X, less) - (Hcond(x, split, X, greater))
        return ig
    
    #entropy of variable
    def H(x, split, X, func, col):
        return P(x, X) * np.log2(P(x, X) * P(not x, X))

    #conditional entropy given split at column
    def Hcond(x, split, X, func, col):
        prob = P(X, less, split, col)
        return -prob * np.log2(prob) -(1-prob) * np.log2(1-prob)

   # probability that random x is higher/lower 
    def P(np.array: X, func, split, col): 
        column = X[:, col]
        selected = [val for val in column if func(val, split)]
        return len(selected)/len(column)



        




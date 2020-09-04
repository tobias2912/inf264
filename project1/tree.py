import numpy as np
class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data
        self.func = None


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

    def __init__(self, X, y):
        root = Node("")

    def get_avg(self, col, X):
        return np.mean(X[:, col])

    def split_X(X, func, best_col)

    def learn(self, X, y, node, impurity_measure='entropy'):
        #if all y is same label, return leaf
        #if all features identical, return most common y
        #try every column to find best gain
        cols, _ = X.shape
        col_gains = []
        for col in cols:
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X)
            col_gains.append(col, gain)
        best_col, best_gain = max(col_gains, key=lambda tup: tup[1])
        print(f"splits on {best_col}")
        #create two branches 
        left = Node("")
        node.left = left 
        right = Node("")
        node.right = right 
        #recursive learn both branches by splitting on the X value
        leftX = split_X(X, less, best_col)
        rightX = split_X(X, greater, best_col)
        learn(leftX, y, left, impurity_measure)
        learn(rightX, y, right, impurity_measure)

    def predict(self, x):
        pass
    

    def less(self, x, y):
        return x<y
        
    def greater(self, x, y):
        return x>y
    #entropy of variable, not used?
    def H(self, x, split, X, func, col):
        pass
        #return self.P(x, X) * np.log2(P(x, X) * self.P(not x, X))

    #conditional entropy given split at column
    def Hcond(self, split, X, func, col):
        prob = self.P(X, self.less, split,  col)
        return -prob * np.log2(prob) -(1-prob) * np.log2(1-prob)

    #information gain
    def IG(self,  col, split, X): 
        #ignores H(y) from calculation
        ig = - self.Hcond( split, X, self.less, col) - (self.Hcond( split, X, self.greater, col))
        return ig

   # probability that random x is higher/lower 
    def P(self, X, func, split, col): 
        column = self.get_column(col, X)
        selected = [val for val in column if func(val, split)]
        return len(selected)/len(column)
    
    def get_column(self, col, X):
        return X[:, col]
    

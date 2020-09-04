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

    root = Node("")

    def __init__(self):
        root = Node("")

    def get_avg(self, col, X):
        c = X[:,col]
        print(c)
        avg = c.mean(axis=0)
        return avg

    def split_X(self, X, func, best_col, split):
        #select rows from matrix where func(best_col) is true
        rows = []
        for row in X:
            if func(row[best_col], split):
                rows.append(row)
        return np.array(rows)

    def less(self, x, y):
        return x<=y
        
    def greater(self, x, y):
        return x>y
    def learn(self, X, y, node, impurity_measure='entropy'):
        if node is None:
            raise Exception("node is None")
        #if all y is same label, return leaf

        #if all features identical, return most common y
        #try every column to find best gain
        _, cols= X.shape
        print(cols)
        col_gains = []
        for col in range(cols):
            print(f"getting avg of column {col}")
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X)
            print(f"information gain: {gain}")
            col_gains.append((col, gain, split_value))
        #select column that gave highest information gain
        best_col, best_gain, split_value = max(col_gains, key=lambda tup: tup[1])
        print(f"splits on {best_col}")
        #create two branches 
        left = Node("")
        node.left = left 
        right = Node("")
        node.right = right 
        #recursive learn both branches by splitting on the X value
        leftX = self.split_X(X, self.less, best_col, split_value)
        rightX = self.split_X(X, self.greater, best_col, split_value)
        #TODO split y
        self.learn(leftX, y, left, impurity_measure)
        self.learn(rightX, y, right, impurity_measure)

    def predict(self, x):
        pass
    

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
    

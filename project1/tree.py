import numpy as np
class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data #split value
        self.column = None # column to split on
        self.func = None
        self.y = None

    def PrintTree(self, depth):
        if self.left is None or self.right is None:
            print ( (depth*"--") + f"Leaf Node( data {self.data}, y {self.y})" )
            return
        print ( (depth*"--") + f"Node( data {self.data}, y {self.y}) children:" )
        self.left.PrintTree(depth+1) 
        self.right.PrintTree(depth+1)  

class Decision_tree:

    root = Node("")

    def __init__(self):
        self.root = Node("")

    def __repr__(self):
        self.root.PrintTree(0)
        return ""

    def print_tree(self):
        self.root.PrintTree(0)

    def get_avg(self, col, X):
        '''returns average of a column in matrix X'''
        c = X[:,col]
        avg = c.mean(axis=0)
        return avg

    def split(self, X, y, func, col, split):
        '''
        select only the rows from X, y where func(X[row, col], split) is true
        '''
        rows = []
        for count, row in enumerate(X):
            if func(row[col], split):
                rows.append(np.append(row, y[count]))
        return np.array(rows)

    def get_common_y(self, y):
        '''return most common value in vector y'''
        return np.bincount(y).argmax()

    def less(self, x, y):
        return x<=y
    def greater(self, x, y):
        return x>y

    def all_identical(self, X):
        '''True if all rows in X are equal'''
        for row in X:
            if np.any(row != X[0]):
                return False
        return True

    def get_best_IG(self, X, impurity_measure):
        '''select column/feature that gives highest information gain'''
        _, cols= X.shape
        col_gains = []
        for col in range(cols):
            print(f"getting avg of column {col}")
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X, impurity_measure)
            print(f"information gain: {gain}")
            col_gains.append((col, gain, split_value))
        #select column that gave highest information gain
        return max(col_gains, key=lambda tup: tup[1])

    def get_best_gini(self, X):
        '''select column/feature that gives Gini'''
        _, cols= X.shape
        col_gains = []
        for col in range(cols):
            print(f"getting avg of column {col}")
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X)
            print(f"information gain: {gain}")
            col_gains.append((col, gain, split_value))
        #select column that gave highest information gain
        return max(col_gains, key=lambda tup: tup[1])
    def learn(self, X, y, node, impurity_measure='entropy'):
        '''
        Build a decision tree with node as a root
        either calls itself recursively, or sets node as a root with a label
        '''
        if node is None:
            raise Exception("node is None")
        #if all y is same label, return leaf
        if np.all(y) or not np.any(y):
            
            print(f"node: {node}, y: {y}")
            node.y = y[0]
            return
        #if all features identical, return most common y
        if self.all_identical(X):
            node.y = self.get_common_y(y)
            return
        #try every column to find best gain
        best_col, best_gain, split_value = self.get_best_IG(X, impurity_measure)
        print(f"splits on {best_col}")
        # inserts data into node
        node.data = split_value
        node.column = best_col
        left = Node("")
        node.left = left 
        right = Node("")
        node.right = right 
        #recursive learn both branches by splitting on the X value
        leftXy = self.split(X,y, self.less, best_col, split_value)
        rightXy = self.split(X,y, self.greater, best_col, split_value)
        leftX = leftXy[:,:4]
        rightX = rightXy[:,:4]
        lefty = leftXy[:,4]
        righty = rightXy[:,4]
        self.learn(leftX, lefty, left, impurity_measure)
        self.learn(rightX, righty, right, impurity_measure)

    
    def predict2(self, node, x):
        assert(isinstance(node, Node))
        column = node.column
        split_value = node.column
        if node.left is None or node.right is None:
            print(node.y)
            return node.y
        if x[column] < split_value:
            return self.predict2(node.left, x)
        else:
            return self.predict2(node.right, x)
   


    """
    def predict(self, x, node):
        '''
        param x: vector
        param node: node to select right or left child
        take left or right in a node and recurively predict until reaching a leaf
        '''
        if node.left is None or node.right is None:
            print(node.y)
        column = node.column
        split_value = node.column
        left = x[column] < split_value
        if left:
            self.predict(x, node.left)
        else:
            self.predict(x, node.right)"""


    #entropy of variable, not used?
    def H(self, x, split, X, func, col):
        pass
        #return self.P(x, X) * np.log2(P(x, X) * self.P(not x, X))

    def Hcond(self, split, X, func, col, impurity_measure):
        '''conditional entropy'''
        prob = self.P(X, self.less, split,  col)
        if impurity_measure=="entropy":
            return -prob * np.log2(prob) -(1-prob) * np.log2(1-prob)
        elif impurity_measure=="gini":
            return -prob * (1-(prob)) -(1-prob) * (1-(1-prob))

    def IG(self,  col, split, X, impurity_measure):
        '''calculate information gain based on entropy given a split''' 
        #ignores H(y) from calculation
        ig = - self.Hcond( split, X, self.less, col, impurity_measure) - (self.Hcond( split, X, self.greater, col, impurity_measure))
        return ig

    def P(self, X, func, split, col): 
        ''' probability that random x is higher/lower'''
        column = self.get_column(col, X)
        selected = [val for val in column if func(val, split)]
        return len(selected)/len(column)
    
    def get_column(self, col, X):
        return X[:, col]
    

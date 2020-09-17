import numpy as np
class Node:

    def __init__(self, data):

        self.left = None
        self.right = None
        self.data = data #split value
        self.majority = None #the majority label of each node
        self.column = None # column to split on
        self.func = None
        self.y = None

    def PrintTree(self, depth):
        if self.left is None or self.right is None:
            print ( (depth*"--") + f"Leaf Node( y {self.y})" )
            return
        print ( (depth*"--") + f"Node( column: {self.column}, data {self.data}, y {self.y}) children:" )
        self.left.PrintTree(depth+1) 
        self.right.PrintTree(depth+1)  

    def __repr__(self):
        if self.right is None and self.left is None:
           return f"leaf with label {self.y}"
        return f"Node with y: {self.y}"

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
        counts = np.bincount(y.astype(int))
        return (np.argmax(counts))
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

    def get_best_IG(self, X, impurity_measure, y):
        '''select column/feature that gives highest information gain'''
        _, cols= X.shape
        col_gains = []
        for col in range(cols):
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X, impurity_measure, y)
            col_gains.append((col, gain, split_value))
        #select column that gave highest information gain
        return max(col_gains, key=lambda tup: tup[1])

    def get_best_gini(self, X):
        '''select column/feature that gives Gini'''
        _, cols= X.shape
        col_gains = []
        for col in range(cols):
            split_value = self.get_avg(col, X)
            gain = self.IG(col, split_value, X)
            col_gains.append((col, gain, split_value))
        #select column that gave highest information gain
        return max(col_gains, key=lambda tup: tup[1])

    def learn(self, X, y, node,  X_pruning, y_pruning, prune=False, impurity_measure='entropy'):
        '''
        Build a decision tree with node as a root
        either calls itself recursively, or sets node as a root with a label
        '''
        if node is None:
            raise Exception("node is None")
        #if all y is same label, return leaf
        if np.all(y) or not np.any(y):
            node.y = y[0]
            return
        #if all features identical, return most common y
        if self.all_identical(X):
            node.y = self.get_common_y(y)
            return
        #try every column to find best gain
        best_col, best_gain, split_value = self.get_best_IG(X, impurity_measure, y)
        # inserts data into node
        node.majority = self.get_common_y(y)
        node.data = split_value
        node.column = best_col
        left = Node("")
        node.left = left 
        right = Node("")
        node.right = right 
        #recursive learn both branches by splitting on the X value
        leftXy = self.split(X,y, self.less, best_col, split_value)
        rightXy = self.split(X,y, self.greater, best_col, split_value)
        if leftXy.size==0 or rightXy.size==0:
            #is leaf
            node.left = None
            node.right = None
            node.y = self.get_common_y(y)
            return
        leftX = leftXy[:,:4]
        rightX = rightXy[:,:4]
        lefty = leftXy[:,4]
        righty = rightXy[:,4]
        self.learn(leftX, lefty, left, X_pruning, y_pruning, prune = prune, impurity_measure = impurity_measure)
        self.learn(rightX, righty, right, X_pruning, y_pruning, prune = prune, impurity_measure = impurity_measure)
        if prune:
            self.prune(X, y, X_pruning, y_pruning, node)
    
    def predict(self, node, x):
        '''
        param x: vector
        param node: node to select right or left child
        take left or right in a node and recurively predict until reaching a leaf
        '''
        assert(isinstance(node, Node))
        column = node.column
        split_value = node.data
        if node.left is None or node.right is None:
            #print("found leaf", node.y)
            return node.y
        if x[column] < split_value:
            #print("going left")
            return self.predict(node.left, x)
        else:
            #print("going right")
            return self.predict(node.right, x)

    def Hcond(self, split, X, func, col, impurity_measure, y):
        '''conditional entropy for one part of the split'''
        prob = self.P(X, func,  split,  col, y)
        if prob == 0 or prob == 1:
            return 0
        if impurity_measure=="entropy":
            return -prob * np.log2(prob) -(1-prob) * np.log2(1-prob)
        elif impurity_measure=="gini":
            return -prob * (1-(prob)) -(1-prob) * (1-(1-prob))

    def IG(self,  col, split, X, impurity_measure, y):
        '''calculate information gain based on entropy given a split''' 
        #ignores H(y) from calculation
        ig = - self.Hcond( split, X, self.less, col, impurity_measure, y) - (self.Hcond( split, X, self.greater, col, impurity_measure, y))
        return ig

    def P(self, X, func, split, col, y): 
        '''splits X into two, and finds probability of y being a 0 in the lower/greater part of split'''
        column = self.get_column(col, X)
        selectedrows = [i for i, val in enumerate(column) if func(val, split)]
        selectedY = [val for i, val in enumerate(y) if i in selectedrows]
        Yzero = [y for y in selectedY if y == 0]
        if len(selectedY)==0:
            return 0.5
        return len(Yzero)/len(selectedY)
   
    def get_column(self, col, X):
        return X[:, col]

    def find_accuracy(self, X_pruning, y_pruning):
        wrong = 1
        correct = 0
        for rownumber, x in enumerate(X_pruning):
            predicted_val = self.predict(self.root, x)
            if predicted_val == y_pruning[rownumber]:
                correct += 1
            else:
                wrong += 1
        accuracy = correct/wrong
        return accuracy

    def prune(self, X, y, X_pruning, y_pruning, node:Node):
        #ignore leafs
        if node.right is None or node.left is None:
            return
        #calculate current accuracy
        accuracy = self.find_accuracy(X_pruning, y_pruning)
        #compare again with node as leaf
        leftbackup = node.left
        rightbackup = node.right
        node.left = None
        node.right = None
        node.y = node.majority
        
        leafaccuracy = self.find_accuracy(X_pruning, y_pruning)
        #if better, replace node with leaf
        if leafaccuracy>=accuracy:
            print(f"new accuracy {leafaccuracy} was better/equal than {accuracy}, pruning...")
        else:
            node.left = leftbackup
            node.right = rightbackup
            node.y = None
            
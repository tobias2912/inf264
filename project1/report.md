# INF264 project 1

## report

# Task 1.1
Implemented the decision tree. We splitted the x-feature on the average of that column. And we splitted the data into test and training data by randomly selected rows with a given seed and a percentage.



# Task 1.2
Changed the entropy calculation to use the gini measure

# Task 1.3
Split training data into training and pruning data. Each node has a field for storing the majority label in subtree. Tries to change node to leaf node and comparing accuracy. If leaf accuracy is worse, then we restore the tree in previous state. 

The pruning happens recursively in the learn method. We're pruning from the leaf node to the top, but it would probably give better performance to prune from the root. 

# Task 1.4


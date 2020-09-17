# INF264 project 1

## report

# Task 1.1
Implemented the decision tree. We splitted the x-feature on the average of that column. We split the data into test and training data by randomly selected rows with a given seed and a percentage. 
The Tree is represented as Node classes, with fields representing child nodes and some data used in building and predicting.



# Task 1.2
Changed the entropy calculation to use the gini measure


# Task 1.3
Split training data into training and pruning data. Each node has a field for storing the majority label in subtree. Tries to change node to leaf node and comparing accuracy. If leaf accuracy is worse, then we restore the tree to its previous state. 

The pruning happens recursively in the learn method, after the tree is built. We're pruning from the leaf node to the top, but it would possibly give better performance to prune from the root. 

# Task 1.4

performance with different settings when testing for accuracy on 30% test data
|         | Pruning | No pruning |
|---------|---------|------------|
| entropy |  97.93  | 98.44      |
| Gini    | 69.25   | 85.78      |

No pruning and entropy gives best result, which we think may be because the test data is very similar to the training data, and it doesnt overfit.

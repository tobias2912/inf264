
Running:
* You need numpy to run the program

* Use the main.py to run

* To change the setting you have to edit this line(line 44 in main):

    <tree.learn(X_train, y_train, tree.root, X_pruning, y_pruning, prune=True, impurity_measure="gini")>
 
    * The default impurity measure is entropy, to use the gini index you have to set define it as: impurity_measure="gini"
    * Can choose between pruning=True or pruning=False





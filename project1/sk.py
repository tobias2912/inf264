import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
import missingno as msno
from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize

# Model Selection
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

# Metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

# Warnings
import warnings as ws
ws.filterwarnings('ignore')

data = pd.read_csv("data_banknote_authentication.txt")
data.head()

X = data.drop('class', axis = 1)
Y = data['class']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 7)


models = []
models.append(('CART', DecisionTreeClassifier()))


def model_selection(x_train, y_train):
    acc_result = []
    auc_result = []
    names = []

    col = ['Model', 'ROC AUC Mean','ROC AUC Std','ACC Mean', 'ACC Std']
    result = pd.DataFrame(columns = col)

    i=0
    for name, model in models:
        kfold = KFold(n_splits = 10, random_state = 7)
        cv_acc_result  = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
        cv_auc_result  = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'roc_auc')

        acc_result.append(cv_acc_result)
        auc_result.append(cv_auc_result)
        names.append(name)

        result.loc[i] = [name, 
                         cv_auc_result.mean(), 
                         cv_auc_result.std(),
                         cv_acc_result.mean(),
                         cv_acc_result.std()]

        result = result.sort_values('ROC AUC Mean', ascending = False)
        i+= 1

    plt.figure(figsize = (10,5))
    plt.subplot(1,2,1)
    sns.boxplot(x = names, y = auc_result)
    plt.title('ROC AUC Score')

    plt.subplot(1,2,2)
    sns.boxplot(x = names, y = acc_result)
    plt.title('Accuracy Score')
    plt.show()

    return(result)

model_selection(x_train, y_train)


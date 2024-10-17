import numpy as np

from sklearn import tree, model_selection
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib
import math

def decision_tree(X_train, y_train, X_test, y_test) -> float:
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


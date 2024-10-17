import matplotlib.pyplot as plt
import pandas

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import kagglehub
import seaborn as sns

import linearsvc
import decisiontree

#importing and cleaning the dataset
dataset = pandas.read_csv('/Users/bhushan/Documents/GitHub/WaterQualityPrediction/Dataset/water_potability.csv')
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
dataset_np = dataset.to_numpy()

#splitting dataset
X = dataset.drop('Potability', axis=1)
y = dataset['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


linearsvc_acc = linearsvc.classifier(X_train, y_train, X_test, y_test)
print(f'Accuracy: {linearsvc_acc:.2f}')

decisionrree_acc = decisiontree.decision_tree(X_train, y_train, X_test, y_test)
print(f'Accuracy: {decisionrree_acc:.2f}')




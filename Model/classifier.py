import matplotlib.pyplot as plt
import pandas

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import kagglehub
import seaborn as sns

#importing and cleaning the dataset
dataset = pandas.read_csv('/Users/bhushan/Documents/GitHub/WaterQualityPrediction/Dataset/water_potability.csv')
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
dataset_np = dataset.to_numpy()


X = dataset.drop('Potability', axis=1)
y = dataset['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#create model using scipy
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#checking accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# print("Classification Report:")
# print(classification_report(y_test, y_pred))






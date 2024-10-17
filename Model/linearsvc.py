from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#create model using scipy
def classifier(X_train, y_train, X_test, y_test) -> float:
    model = LinearSVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy






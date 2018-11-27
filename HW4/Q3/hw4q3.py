## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize


######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
# XXX
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True, random_state=100)



# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lrc = LinearRegression().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX
print("Linear regression:")
print("Training:", accuracy_score(y_train, lrc.predict(X_train).round()))
print("Testing:", accuracy_score(y_test, lrc.predict(X_test).round()), "\n")



# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX
mlp = MLPClassifier().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("MLP:")
print("Training:", accuracy_score(y_train, mlp.predict(X_train).round()))
print("Testing:", accuracy_score(y_test, mlp.predict(X_test).round()), "\n")



# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX
rfc = RandomForestClassifier().fit(X_train, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("Random forest:")
print("Training:", accuracy_score(y_train, rfc.predict(X_train).round()))
print("Testing:", accuracy_score(y_test, rfc.predict(X_test).round()), "\n")

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX
parameters = {'n_estimators': [10, 30, 50, 70, 100], 'max_depth': [2, 4, 6, 8, 10]}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10).fit(X_train, y_train)
print("Random forest tuning:")
print("Best parameters:", clf.best_params_)
print("Best score:", clf.best_score_, "\n")



# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
X_train_std = StandardScaler().fit(X_train, y_train).transform(X_train)
X_test_std = StandardScaler().fit(X_test, y_test).transform(X_test)
svc = SVC().fit(X_train_std, y_train)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("SVM:")
print("Training:", accuracy_score(y_train, svc.predict(X_train_std).round()))
print("Testing:", accuracy_score(y_test, svc.predict(X_test_std).round()), "\n")

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX
params = {'C': [0.001, 0.01, 0.1], 'kernel': ('rbf', 'linear')}
svclf = GridSearchCV(SVC(), params, cv=10).fit(X_train_std, y_train)
print("SVM tuning:")
print("Best parameters:", svclf.best_params_)
print("Best score:", svclf.best_score_, "\n")

best_params = {'C': [0.1,], 'kernel': ('rbf',)}
best_clf = GridSearchCV(SVC(), best_params, cv=10).fit(X_train_std, y_train)
print("Mean training score:", best_clf.cv_results_['mean_train_score'])
print("Mean testing score:", best_clf.cv_results_['mean_test_score'])
print("Mean fit time:", best_clf.cv_results_['mean_fit_time'])


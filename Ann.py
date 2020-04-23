# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:10:03 2020

@author: Saad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


bankdata2 = pd.read_csv("trainingCommonVoicePreProcessedWithAgeGroupsRemovingOther.csv")

X = bankdata2.iloc[:, :-4].values
y = bankdata2.iloc[:, [34,36,37]].values


# # Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y0 = LabelEncoder()
labelencoder_y1 = LabelEncoder()
labelencoder_y2 = LabelEncoder()
y[:,0] = labelencoder_y0.fit_transform(y[:,0])
y[:,1] = labelencoder_y1.fit_transform(y[:,1])
y[:,2] = labelencoder_y2.fit_transform(y[:,2])
y = y.astype(float)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 34))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train[:,0], batch_size = 10, epochs = 50)



import numpy 
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred =numpy.multiply(y_pred>0.5, 1) 



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test[:,0], y_pred)

from sklearn.metrics import accuracy_score, classification_report
print(cm)
print(classification_report(y_test[:,0],y_pred))
print("accuracy:",accuracy_score(y_test[:,0],y_pred,normalize=True))


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu', input_dim = 34))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [50, 60],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train[:,0])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
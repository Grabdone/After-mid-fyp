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

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y0 = LabelEncoder()
labelencoder_y1 = LabelEncoder()
labelencoder_y2 = LabelEncoder()
y[:,0] = labelencoder_y0.fit_transform(y[:,0])
y[:,1] = labelencoder_y1.fit_transform(y[:,1])
y[:,2] = labelencoder_y2.fit_transform(y[:,2])
y = y.astype('int')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train[:,0])
X_test = lda.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train[:,0])

# Predicting the Test set results
y_pred = classifier.predict(X_test)

    

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test[:,0], y_pred)

from sklearn.metrics import accuracy_score, classification_report
print(cm)
print(classification_report(y_test[:,0],y_pred))
print("accuracy:",accuracy_score(y_test[:,0],y_pred,normalize=True))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train,  y_train[:,0]

plt.xlim(X_set.min(), X_set.max())
plt.ylim(y_set.min(), y_set.max())
plt.scatter(X_set, y_set)
plt.title('LDA (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
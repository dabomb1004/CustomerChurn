# -*- coding: utf-8 -*-
"""class 6

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/186oZ81KY0TsMhb7QVY93pLUCdWmaaf8C
"""

import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('C6-Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

dataset

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X_new = ct.fit_transform(X)
X_new[0,]

labelencoder_X = LabelEncoder()
X_new[:, 4] = labelencoder_X.fit_transform(X_new[:,4])
X_new[0:10,]

#split data into training and test 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state = 0)

X_train

X_test

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

#test how good is this model 

y_train_pred = classifier.predict(X_train)
y_train_pred = (y_train_pred > 0.5).flatten()*1

#confusion matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_train_pred)

print(cm) 

import numpy as np 
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_train, y_train_pred.flatten()*1))
print(rms)

rms = sqrt(sum((y_train - y_train_pred)**2)/len(y_train))
print(rms)

#test how good is this model 

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).flatten()*1

#confusion matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm) 

import numpy as np 
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred.flatten()*1))
print(rms)

rms = sqrt(sum((y_test - y_pred)**2)/len(y_test))
print(rms)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_probability = classifier.predict(sc.transform(np.array([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_probability > 0.5)
new_prediction

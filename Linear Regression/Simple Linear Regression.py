# Simple Linear Regression
# Problem Statement : Based on CGPA we have to predict Salary Package

# # Mount Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, explained_variance_score

# Load Dataset
data = pd.read_csv('/content/drive/MyDrive/Linear Regression/placement.csv')
data.head(5)

# Here we will not use any csv file, we will create our own dataset
# Data Preparation
# CGPA column (As GPA in range 1 to 10)
x_col = np.random.uniform(low=1.0, high=10.0, size=(10000,))
x_col = np.round(x_col, 2)

# Package Column (As Package can be anything from 10 LPA to 50 LPA)
y_col = np.random.uniform(low=1.0, high=10.0, size=(10000,))
y_col = np.round(y_col, 2)

# Now convert it into Pandas DataFrame
df = pd.DataFrame({'CGPA':x_col, 'Package': y_col})
df.head(5)

# Split Dataset into Training and Testing
# Now we have to extract X and Y column to create a data_set split
# CSV dataset
# X = data.iloc[:, :1].values
# y = data.iloc[:, 1:].values

# Our dataset
X = df.iloc[:, :1].values
y = df.iloc[:, 1:].values

# Then create a dataset for training (80% : 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# # Feature Scaling (Only if necessary)
# from sklearn.preprocessing import StandardScaler

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Training the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('CGPA vs Package (Training set)')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.show()

# Testing or Predicting the Test set results
y_pred = lr.predict(X_test)
# print(y_pred)

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('CGPA vs Package (Test set)')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.show()

# Calculate Accuracy
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error", mse)
print("Mean Absolute Error", mae)
print("Explained Variance Score", evs)
print("R2 Score", r2)

# Intercept and Coefficient
m = lr.coef_
c = lr.intercept_
print(m, c)

# Top 5 x_test and y_test
print(X_test[:5])
print(y_test[:5])

# Predictions

# Check the actiula values of 3rd data point form our test set
point_x = X_test[3]
point_y = y_test[3]
print("Actual value and prediction on this 3rd data point is: ", point_x, point_y)

# Prediction on the same data point by calculation (y = mx + c)
prediction = m * X_test[3] + c
prediction = np.round(prediction, 2)
print("Predicted package by our model : ", prediction)
# -*- coding: utf-8 -*-
"""Multiple_Disease_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cc1N7zOCspR2Rz4BT0ereXZuc4D0zcnR

**Importing Required Libraries (Dependencies)**
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score

"""**Data Collection and Analysis**"""

#Uploading and reading the CSV file
multiple_disease_dataset=pd.read_csv('/content/Diseasedata.csv')

#checking for NaN values
print(multiple_disease_dataset.isnull().sum())

"""# New Section"""

#Printing first five lines of the file (optional)
multiple_disease_dataset.head()

#Counting no.of rows and columns (optional)
multiple_disease_dataset.shape

#Descriptions of values of every features (optional)
multiple_disease_dataset.describe()

#No.of times a disease has been taken in the data set (optional)
multiple_disease_dataset['Disease'].value_counts()

#Seperating the data and disease
X = multiple_disease_dataset.drop(columns='Disease',axis=1)
Y = multiple_disease_dataset['Disease']

#Printing the values of X and Y (optional)
print(X)
print(Y)

"""**Splitting, Training and Testing the data**"""

#Splitting the data into training and testing with size 80% and 20% respectively
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,stratify=Y,random_state=42)

#Training the data using the Suppot Vector Machine model
classifier_svm=svm.SVC(kernel='linear')
classifier_svm.fit(X_train,Y_train)

#Training the data using the Decision Tree Model
classifier_decision_tree=tree.DecisionTreeClassifier()
classifier_decision_tree.fit(X_train,Y_train)

#Training the data using the Random Forest Model
classifier_random_forest=ensemble.RandomForestClassifier()
classifier_random_forest.fit(X_train,Y_train)

"""**Model Evaluation**"""

#Predicting the outcomes for the trained data using svm model
X_train_prediction_S=classifier_svm.predict(X_train)

#Measuring accuracy score for training data using svm model
train_data_accuracy_S=accuracy_score(X_train_prediction_S,Y_train)
print(train_data_accuracy_S)

#Predicting the outcomes for the trained data using decision tree model
X_train_prediction_D=classifier_decision_tree.predict(X_train)

#Measuring accuracy score for training data using decision tree model
train_data_accuracy_D=accuracy_score(X_train_prediction_D,Y_train)
print(train_data_accuracy_D)

#Predicting the outcomes for the trained data using random forest model
X_train_prediction_R=classifier_random_forest.predict(X_train)

#Measuring accuracy score for training data using random forest model
train_data_accuracy_R=accuracy_score(X_train_prediction_R,Y_train)
print(train_data_accuracy_R)

#Predicting the outcomes for the test data using svm model
X_test_prediction_S=classifier_svm.predict(X_test)

#Measuring the accuracy score for the test data using svm model
test_data_accuracy_S=accuracy_score(X_test_prediction_S,Y_test)
print(test_data_accuracy_S)

#Predicting the outcomes for the test data using decision tree model
X_test_prediction_D=classifier_decision_tree.predict(X_test)

#Measuring the accuracy score for the test data using svm model
test_data_accuracy_D=accuracy_score(X_test_prediction_D,Y_test)
print(test_data_accuracy_D)

#Predicting the outcomes for the test data using random forest model
X_test_prediction_R=classifier_random_forest.predict(X_test)

#Measuring the accuracy score for the test data using random forest model
test_data_accuracy_R=accuracy_score(Y_test,X_test_prediction_R)
print(test_data_accuracy_R)

"""**Predicting System**"""

# Define symptom list (all feature names)
all_symptoms = list(X.columns)

# Taking symptoms as input from the user
input_symptoms = {'Cough', 'NA', 'NA', 'NA', 'Sweating'}

# Converting the input to feature vector
input_data = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]

#Changing the data into numpy array data frame
input_data_as_numpy=np.asarray(input_data)

#Reshaping the data
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

#Predicting the Output
prediction=classifier_random_forest.predict(input_data_reshaped)

#Printing the output as Yes or No
print(prediction[0])
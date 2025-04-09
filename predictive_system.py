# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:36:10 2025

@author: mallepalli gautham
"""

import numpy as np
import pickle

#load the model
loaded_model=pickle.load(open('D:/project\GAUTHAM_PRO1/trained_model.sav','rb'))

# Define symptom list (all feature names)
all_symptoms = ["Fever", "Shortness of breath", "Cough", "Chest pain", "Nausea", "Vomiting", "Lightheadedness", "Sweating", "Sudden weakness", "Numbness", "Confusion", "Headache", "Lump", "Weight loss", "Fatigue", "Bleeding", "Seizures", "Swelling", "Conjunctivitis", "Diarrhea", "Liver Damage", "Cancer", "Stiff Neck", "Pain in upper abdomen"]

# Taking symptoms as input from the user
input_symptoms = {'Cough', 'NA', 'NA', 'NA', 'Sweating'}

# Converting the input to feature vector
input_data = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]

#Changing the data into numpy array data frame
input_data_as_numpy=np.asarray(input_data)

#Reshaping the data
input_data_reshaped=input_data_as_numpy.reshape(1,-1)

#Predicting the Output
prediction=loaded_model.predict(input_data_reshaped)

#Printing the output
print (f"The person is having {prediction[0]} disease")
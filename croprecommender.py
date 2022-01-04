from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pyinputplus as pyip


cropdata = pd.read_csv("Crop_recommendation.csv")
crop_summary = pd.pivot_table(cropdata,index=['label'],aggfunc='mean')
print()
print("CROP MEAN CHART")
print()
print(crop_summary)


print('''
This program gives the most suitable crop from 
'rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans' 'mungbean' 
'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes''watermelon' 
'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton''jute' 'coffee' 
based on the given features i.e. N, P, K, Temperature, Humidity, pH, Rainfall
''')
print()
print("Enter Features")
Ka = pyip.inputInt("K : ")
Ni = pyip.inputInt("N : ")
Po = pyip.inputInt("P : ")
humidity = pyip.inputFloat("Humidity : ")
ph = pyip.inputFloat("pH : ")
rainfall = pyip.inputFloat("Rainfall : ")
temperature = pyip.inputFloat("Temperature : ")


print()
print("Using random forest")
print()
print("Training from Crop_recommendation.csv")

features = cropdata[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = cropdata['label']

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(features,target)

x_test = [[ Ni, Po,Ka,temperature, humidity, ph, rainfall]]

print()
print("predicting.....")
print()

predicted_value = RF.predict(x_test)
print("Best crop Suitable for this condition is : ",predicted_value[0])

print()
from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pyinputplus as pyip


print('''
This program gives the most suitable crop from 
'rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans' 'mungbean' 
'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes''watermelon' 
'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 'cotton''jute' 'coffee' 
based on the given features i.e. N, P, K, Temperature, Humidity, pH, Rainfall
''')

cropdata = pd.read_csv("Crop_recommendation.csv")
features = cropdata[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = cropdata['label']
acc = []
model = []
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(features,target)

Ni = pyip.inputInt("N : ")
Po = pyip.inputInt("P : ")
Ka = pyip.inputInt("K : ")
temperature = pyip.inputFloat("Temperature : ")
humidity = pyip.inputFloat("Humidity : ")
ph = pyip.inputFloat("pH : ")
rainfall = pyip.inputFloat("Rainfall : ")

x_test = [[ Ni, Po,Ka,temperature, humidity, ph, rainfall]]

predicted_value = RF.predict(x_test)

print("Best crop Suitable for this condition is : ",predicted_value[0])
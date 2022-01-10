from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pyinputplus as pyip
from sklearn.naive_bayes import GaussianNB


cropdata = pd.read_csv("Crop_recommendation.csv")
crop_summary = pd.pivot_table(cropdata,index=['label'],aggfunc='mean')
print("--------------------------------------------------------------------------------------------------------------------")
print("CROP MEAN CHART ( Best Features required for each crops. )")
print("--------------------------------------------------------------------------------------------------------------------")
print(crop_summary)
print("--------------------------------------------------------------------------------------------------------------------")
print('''
This program gives the most suitable crop from 'rice' 'maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans' 'mungbean' 
'blackgram' 'lentil' 'pomegranate' 'banana' 'mango' 'grapes''watermelon' 'muskmelon' 'apple' 'orange' 'papaya' 'coconut' 
'cotton''jute' 'coffee' based on the given features i.e. N, P, K, Temperature, Humidity, pH, Rainfall
''')
print()

print("Enter Features : ")
Ka = pyip.inputInt("K : ")
Ni = pyip.inputInt("N : ")
Po = pyip.inputInt("P : ")
humidity = pyip.inputFloat("Humidity : ")
ph = pyip.inputFloat("pH : ")
rainfall = pyip.inputFloat("Rainfall : ")
temperature = pyip.inputFloat("Temperature : ")

features = cropdata[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = cropdata['label']
x_test = [[ Ni, Po,Ka,temperature, humidity, ph, rainfall]]
print(x_test)

print("--------------------------------------------------------------------------------------------------------------------")
print("Using random forest")
print()
print("Training from Crop_recommendation.csv")


RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(features,target)

print()
print("predicting.....")
print()

Rpredicted_values = RF.predict(x_test)
print("Best crop Suitable for this condition is : ",Rpredicted_values[0])

print("--------------------------------------------------------------------------------------------------------------------")


print("--------------------------------------------------------------------------------------------------------------------")
print("Using Naïve Bayes")
print()
print("Training from Crop_recommendation.csv")

NaiveBayes = GaussianNB()
NaiveBayes.fit(features,target)

print()
print("predicting.....")
print()

Npredicted_values = NaiveBayes.predict(x_test)

print("Best crop Suitable for this condition is : ",Npredicted_values[0])
print("--------------------------------------------------------------------------------------------------------------------")

if (str(Npredicted_values[0]) == str(Rpredicted_values[0])):
    print("Suitable crop is : ", Npredicted_values[0])

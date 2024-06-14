# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#import missingno as msno # To visualize missing value
#import plotly.graph_objects as go # To Generate Graphs
#import plotly.express as px # To Generate box plot for statistical representation
#%matplotlib inline
df = pd.read_csv("D://proj//heart data.csv")
df.dtypes

age_mapping = {
    '18-24': 1,
    '25-29': 2,
    '30-34': 3,
    '35-39': 4,
    '40-44': 5,
    '45-49': 6,
    '50-54': 7,
    '55-59': 8,
    '60-64': 9,
    '65-69': 10,
    '70-74': 11,
    '75-79': 12,
    '80 or older': 13
}

# Apply the mapping to the 'AgeCategory' column
df['AgeCategory'] = df['AgeCategory'].map(age_mapping)

bmi_upper_limit = 42.505  # Example threshold for BMI outliers
df.loc[df['BMI'] > bmi_upper_limit, 'BMI'] = df['BMI'].mean()

# Replace outlier values in 'PhysicalHealth' with the mean
physical_health_upper_limit = 5.0  # Example threshold for PhysicalHealth outliers
df.loc[df['PhysicalHealth'] > physical_health_upper_limit, 'PhysicalHealth'] = df['PhysicalHealth'].mean()

# Replace outlier values in 'PhysicalHealth' with the mean
Mental_health_upper_limit = 7.5  # Example threshold for PhysicalHealth outliers
df.loc[df['MentalHealth'] > Mental_health_upper_limit, 'MentalHealth'] = df['MentalHealth'].mean()

# Replace outlier values in 'SleepTime' with the mean
sleep_time_upper_limit = 11.0  # Example threshold for SleepTime outliers
df.loc[df['SleepTime'] > sleep_time_upper_limit, 'SleepTime'] = df['SleepTime'].mean()

df = df.replace({'HeartDisease': {'Yes': 1, 'No': 0}})

data_num=df.select_dtypes([int,float])#all numerical columns 
data_cat=df.select_dtypes(object)#all categorical columns

from sklearn.preprocessing import LabelEncoder

binary_categorical_columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'GenHealth','PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
for i in binary_categorical_columns:
    data_cat[i] = LabelEncoder().fit_transform(data_cat[i])
    
data_cat.head()

race_dummies = pd.get_dummies(data_cat['Race'], prefix='Race')
diabetic_dummies = pd.get_dummies(data_cat['Diabetic'], prefix='Diabetic')

data_cat = pd.concat([data_cat, race_dummies, diabetic_dummies], axis=1)

data_cat.drop(['Race', 'Diabetic'], axis=1, inplace=True)

data_cat.head()

data_new=pd.concat([data_num,data_cat],axis=1)

data_new.head()

L=['BMI','PhysicalHealth','MentalHealth','AgeCategory','SleepTime']
print(L)


from sklearn.preprocessing import StandardScaler
for i in L:
    sd=StandardScaler()
    data_new[i]=sd.fit_transform(data_new[[i]])
X = data_new[['BMI', 'PhysicalHealth', 'MentalHealth','AgeCategory','SleepTime','Smoking','AlcoholDrinking','Stroke','DiffWalking']]
y = data_new[['HeartDisease']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 
logreg = LogisticRegression()
logreg.fit(X_train, y_train.values.ravel())

pickle.dump(logreg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
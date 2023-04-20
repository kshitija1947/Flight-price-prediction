#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('clean_dataset.csv')

#Deleting unwanted columns
dataset.drop(["Unnamed: 0"],axis=1,inplace=True)
dataset.drop(["flight"],axis=1,inplace=True)



#Converting non-numerical varible to numerical
from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder
dataset["airline"]=l1.fit_transform(clean_data["airline"])
dataset["source_city"]=l1.fit_transform(clean_data["source_city"])
dataset["departure_time"]=l1.fit_transform(clean_data["departure_time"])
dataset["stops"]=l1.fit_transform(clean_data["stops"])
dataset["arrival_time"]=l1.fit_transform(clean_data["arrival_time"])
dataset["destination_city"]=l1.fit_transform(clean_data["destination_city"])
dataset["class"]=l1.fit_transform(clean_data["class"])
#clean_data["stops"]=l1.fit_transform(clean_data["stops"])




x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]




#Splitting Training and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 1)


#Feature scaling (Standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
#print(X_train)

X_test = sc.transform(x_test)
#print(X_test)



#Model training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train,y_train)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5,2,4,2,0,5,1,2.25,1]]))
'''

#!/usr/bin/python3.5

#ANN model applied to Corporate Investment Grade Bonds
#Purely for Demonstration purposes 

import numpy as np
import pandas as pd 
from datetime import datetime 
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def main():
 
	#input raw bond data
	raw_Bond_data = pd.read_csv('CSV Bond Data File Path')

	#Remove all entries with missing data
	#Done for model demonstration purposes
	#Optional
	bond_data = raw_Bond_data.dropna()

	#Convert 'Maturity' and 'S&P Init Rtg Dt' to datetime
	bond_data['Maturity'] = pd.to_datetime(bond_data['Maturity'])
	bond_data['S&P Init Rtg Dt'] = pd.to_datetime(bond_data['S&P Init Rtg Dt'])

	#Convert 'Maturity' and 'S&P Init Rtg Dt' to ordinal numbers
	bond_data['Maturity'] = bond_data['Maturity'].apply(datetime.toordinal)
	bond_data['S&P Init Rtg Dt'] = bond_data['S&P Init Rtg Dt'].apply(datetime.toordinal)

	#Convert Categorical Ratings 
	labelencoder_rate = LabelEncoder()
	bond_data['S&P Rating']=labelencoder_rate.fit_transform(bond_data['S&P Rating'])

	labelencoder_init_rate = LabelEncoder()
	bond_data['S&P Init Rtg'] = labelencoder_init_rate.fit_transform(bond_data['S&P Init Rtg'])

	#Create Binary field to depict change in bond rating
	bond_data['Binary'] = bond_data['S&P Init Rtg'] < bond_data['S&P Rating']
	bond_data['Binary'] = bond_data.Binary.astype(int)

	#Convert 'Maturity Type'
	labelencoder_mat_type = LabelEncoder()
	bond_data['Maturity Type']=labelencoder_mat_type.fit_transform(bond_data['Maturity Type'])

	#Convert 'S&P Outlook'
	labelencoder_outlook = LabelEncoder()
	bond_data['S&P Outlook'] = labelencoder_outlook.fit_transform(bond_data['S&P Outlook'])
	bond_data['S&P Outlook']

	#independent variables
	# dependent variables 
	#'Cpn', 'Maturity', 'Maturity Type',
    #'90D Volatility', 'Yld to Mty (Mid)', 'Rev Growth', 'Amount Issued',
    #'Amt Out', 'CDS Recovery Rate', 'CDS Spread (5yr Mid)',
    #'Earnings Assets / Interest Bearing Liabililties', 'S&P Init Rtg Dt', 
    #'S&P Outlook',
	x = bond_data.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,16]].values

	#Dependent variable 
	# 'S&P Rating'
	y = bond_data.iloc[:,18].values

	#split into training and test partitions
    # .30 = test, .70 = train
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state = 0)

	#feature scaling 
	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.fit_transform(x_test)

	#intialize ANN
	classifier = Sequential()

	# Adding the input layer and the first hidden layer
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))

	# Adding the second hidden layer
	classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))

	# Adding the output layer
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

	# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	# Fitting the ANN to the Training set
	#batch_size = after number of observations to update wieghts
	classifier.fit(x_train, y_train, batch_size = 10, epochs = 1500)	

	# Predicting the Test set results
	y_pred = classifier.predict(x_test)
	print(y_pred)

	# Making the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred.round())
	print(cm)

main()
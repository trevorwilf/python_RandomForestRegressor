
import pandas as pd
import os
import csv
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import math

# import the data
iowa_file_path = './data/train.csv'
iowa_file_path2 = './data/test.csv'

home_data = pd.read_csv(iowa_file_path)
#home_data2 = pd.read_csv(iowa_file_path2)
#home_data = pd.concat([home_data1, home_data2])

# trying to predict saleprice
y_train = home_data.SalePrice

# subset the features out
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x_train = home_data[features]

#split the training set into train and validate - seems unnecessary as teh data set is already split
train_X, val_X, train_y, val_y = train_test_split(x_train, y_train, random_state=1)

#create model
iowa_model_rf = RandomForestRegressor(random_state=1)

#fit the model
iowa_model_rf.fit(train_X, train_y)

# test model
predictions = iowa_model_rf.predict(val_X)

#get error
iowa_model_rf_mae = round(mean_absolute_error(val_y, predictions), ndigits= 2)
iowa_model_rf_rmse = round(math.sqrt(mean_absolute_error(val_y, predictions)), ndigits=2)

print(f'The MAE is: {iowa_model_rf_mae}')
print(f'The RMSE is: {iowa_model_rf_rmse}')
#Loading required libraries

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas_profiling
import time
import configparser as configParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


start_time = time.time()

#getting data from configuration file
configParser =configParser.RawConfigParser()
configFilePath = 'configuration.txt'
configParser.read(configFilePath)
input_path=configParser.get('input', 'path')
output_path=configParser.get('output', 'path')
input_file=configParser.get('input','input_file')
target_variable=configParser.get('input','target_variable')

#reading the dataset
train = pd.read_csv(input_path+input_file+".csv",encoding = 'ISO-8859-1')

#removing the unique column fields:
for i in train.columns:
    if (len(train[i]) == train[i].nunique()):
        train.drop([i], axis=1, inplace=True)

#converting few numerical values into objects based on unique values
for i in train.columns:
    if (train.dtypes[i] != np.object):
        if(train[i].nunique()==2 or train[i].nunique()<=10):
            train[i] = train[i].astype('object')

#missing values normalization:
for i in train.columns:
    if(train.dtypes[i]==np.object):
        train[i].fillna(train[i].mode()[0], inplace=True)
    if(train.dtypes[i]!=np.object):
        train[i].fillna(train[i].mean(),inplace=True)

#label encoding of the objects
for i in train.columns:
    if train.dtypes[i]==np.object:
        lb_make = LabelEncoder()
        train[i] = lb_make.fit_transform(train[i])

#assigning values to x and y based on target variable:
x=train.drop(target_variable,axis=1).values
label=train.drop(target_variable,axis=1).columns.values
y=train[target_variable].values

#Spliting training and testing data

predictors_train, predictors_test, target_train, target_test = train_test_split(x,y, test_size=0.3, random_state=0)

#Training the model using training data

mlr_mod = sm.OLS(target_train,predictors_train).fit()

mlr_mod.summary()

#Evaluating model using testing data

pred = mlr_mod.predict(predictors_test)

met_df = pd.DataFrame(columns=["Model_name", "R-Square", "RMSE", "MAE"])
r2 = r2_score(target_test, pred)
rmse = np.sqrt(mean_squared_error(target_test, pred))
mae = mean_absolute_error(target_test, pred)
met_df.loc[len(met_df)] = ["MLR", r2, rmse, mae]

print(met_df)

#Calculate run time
print("\n Time Elapsed: {:.2f}s".format(time.time() - start_time))

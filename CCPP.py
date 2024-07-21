# to handle data in form of rows and columns
import pandas as pd

# Numerical libraries
import numpy as np

# importing ploting libraries
import matplotlib.pyplot as plt

#importing seaborn for statistical plots
import seaborn as sns

#implements serialization
import pickle

data = pd.read_csv(r"C:\project\HLO1.csv", header=0 , names = ['AT','V','AP','RH','PE'])
print(data.isnull().sum())
print(data.head())
print(data.describe().T)
plt.scatter(data['AT'],data['PE'])
plt.scatter(data['V'],data['PE'])
plt.scatter(data['AP'],data['PE'])
plt.scatter(data['RH'],data['PE'])
sns.pairplot(data,diag_kind = 'hist')
x = data.drop(['PE'], axis=1)
y = data['PE']
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
xtrain.shape
print(xtrain.shape)
xtest.shape
print(xtest.shape)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# Initializing the models
LRmodel=LinearRegression()
DTmodel=DecisionTreeRegressor()
RFmodel=RandomForestRegressor()
from sklearn.linear_model import LinearRegression
LRmodel = LinearRegression()
LRmodel.fit(xtrain, ytrain)
print(LRmodel.fit(xtrain, ytrain))
from sklearn.tree import DecisionTreeRegressor
DTRmodel = DecisionTreeRegressor()
DTRmodel.fit(xtrain, ytrain)
print(DTRmodel.fit(xtrain, ytrain))
from sklearn.ensemble import RandomForestRegressor
RFmodel = DecisionTreeRegressor()
RFmodel.fit(xtrain, ytrain)
print(RFmodel.fit(xtrain, ytrain))


# Linear Regression
from sklearn.linear_model import LinearRegression
# Initializing the model
LRmodel = LinearRegression()
# Train the data with Linear Regreesion model
LRmodel.fit(xtrain, ytrain)
LinearRegression()
LRpred=LRmodel.predict(xtest)
# Importing R Square Library
from sklearn.metrics import r2_score
# Checking for accuracy score with actual data and predicted data
LRscore=r2_score(ytest, LRpred)
LRscore
print(LRscore)


# DECISION TREE Regression
from sklearn.tree import DecisionTreeRegressor
# Initializing the model
DTRmodel = DecisionTreeRegressor()
# Train the data with Linear Regreesion model
DTRmodel.fit(xtrain, ytrain)
DecisionTreeRegressor()
DTRpred=DTRmodel.predict(xtest)
# Importing R Square Library
DTRscore=r2_score(ytest, DTRpred)
DTRscore
print(DTRscore)

# Linear Regression
from sklearn.ensemble import RandomForestRegressor
# Initializing the model
RFmodel = RandomForestRegressor()
RFmodel.fit(xtrain, ytrain)
RandomForestRegressor()
RFpred=RFmodel.predict(xtest)
# Importing R Square Library
RFscore=r2_score(ytest, RFpred)
RFscore
print(RFscore)

pickle.dump(RFmodel, open('CCPP.pkl','wb'))

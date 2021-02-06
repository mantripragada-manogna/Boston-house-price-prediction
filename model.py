# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
#import xgboost as xgb
#from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import pickle
import joblib
import requests
import json
#Importing the data and checking the top 10 rows of the training data frame
train_df = pd.read_csv('HousingData_train.csv')

#Important features 
features = ['LSTAT', 'RM', 'CRIM', 'DIS', 'TAX', 'PTRATIO']
train_df_updated = train_df[features]

#Handling nulls
nullcols = ['LSTAT']
train_df_updated['LSTAT'].fillna(train_df_updated['LSTAT'].mean(), inplace = True)
train_df_updated['CRIM'].fillna(train_df_updated['CRIM'].mean(), inplace = True)

# Splitting the dataset into the Training set and Validation set
X = train_df_updated
y = train_df['MEDV']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=5)


# Fitting Simple Linear Regression to the Training set

randforest_model = RandomForestRegressor(n_estimators = 310, min_samples_leaf= 1, max_features= 'log2',bootstrap= False)
print(X_train.isnull().sum())
randforest_model.fit(X_train, y_train)
y_val_pred = randforest_model.predict(X_val)
rmse = (np.sqrt(mean_squared_error(y_val, y_val_pred)))
r2 = r2_score(y_val, y_val_pred)
print('RMSE score is: {} and R2 is {}'.format(rmse, r2))

#Importing test data
X_test = pd.read_csv('HousingData_test.csv')
X_test = X_test[features]

#Handling missing data
X_test['LSTAT'].fillna(X_test['LSTAT'].mean(), inplace = True)
print(X_test.isnull().sum())


# Predicting the Test set results
y_pred = randforest_model.predict(X_test)

# Saving model to disk
pickle.dump(randforest_model, open('randomforest.pkl','wb'))
#save_file_name = 'randomforest.pkl'
#joblib.dump(randforest_model, save_file_name)
print('saved pipeline')

# Loading model to compare the results
#model_reloaded = joblib.load('randomforest.pkl')


model = pickle.load(open('randomforest.pkl','rb'))
pred = randforest_model.predict(X_test)
print(pred)
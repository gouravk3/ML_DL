import pandas as pd
import numpy as np
import re
from sklearn import metrics
dataset=pd.read_csv("C:\\Users\\GK\Desktop\\ML\\Stock_price_prediction\\dataset.csv")
dataset = dataset.replace({'Change %' : '%'} ,'' , regex=True)
dataset['Change %'] = dataset['Change %'].astype(float)
dataset = dataset.replace({'Vol.' : 'M'} ,'' , regex=True)
dataset['Vol.'] = dataset['Vol.'].astype(float)
dataset.describe()
x  = dataset[['Open','High','Low','Vol.' ,'Change %']]
y = dataset['Close']
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x ,y , random_state = 0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(x_test)
dframe=pd.DataFrame(y_test,predicted)
dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})
print(dfr)
dfr.head(25)
from sklearn.metrics import confusion_matrix, accuracy_score
regressor.score(x_test,y_test)
import math
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predicted))
print('Mean Squared  Error:',metrics.mean_squared_error(y_test,predicted))
print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predicted)))
print('Accuracy = ' ,regressor.score(x_test,y_test)*100, '%')

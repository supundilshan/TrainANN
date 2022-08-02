from sklearn.model_selection import train_test_split

# Regression Example With Boston Dataset: Baseline
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# load dataset
Train_dataframe = read_csv("Input/Pure_Data.csv")
Train_dataset = Train_dataframe.values
X_train = Train_dataset[:, 3:10]
Y_train = Train_dataset[:, 2]

# ===== Normalize Data set =====
# create scaler
# scaler = MinMaxScaler()
# # fit and transform in one step
# X_train = scaler.fit_transform(X_train)

# ===== Standerdize Data set =====
# created scaler
scaler = StandardScaler()
# fit scaler on training dataset
scaler.fit(X_train)
# transform training dataset
X_train = scaler.transform(X_train)

# define base model
def Regression_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# # evaluate model
estimator = KerasRegressor(build_fn=Regression_model, epochs=250, batch_size=30, verbose=1)
# # 430,10 and 440,10 and 450,10 are best
#
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
#
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# # Precict vaues
estimator.fit(X_train, Y_train)
#
prediction = estimator.predict(X_train)

# evaluate model with standardized dataset
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=Regression_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=5)
# results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# # Precict vaues
# pipeline.fit(X_train, Y_train)
# prediction = pipeline.predict(X_train)

MSE =mean_squared_error(Y_train, prediction)
print("MSE : ", MSE)

# get MAPE
MAPE = mean_absolute_percentage_error(Y_train, prediction)
print("MAPE : ", MAPE)

# get R-squared
r2 = r2_score(Y_train, prediction)
print("R-squared : ", r2)
#
# prediction = np.array(read_csv("PredOLS.csv")).flatten()

#
plt.plot(prediction, Y_train, '.', color='black')
# create scatter plot
m, b = np.polyfit(prediction, Y_train, 1)
# m = slope, b=intercept
plt.plot(prediction, m*prediction + b, color='red')
plt.xlabel("Predicted values")
plt.ylabel("Actual Values")

# get MSE manualy Method-1
# train_error = np.abs(Y - prediction)
# sqrt_train_error = train_error**2
# sumOf_sqrt_train_error = np.sum(sqrt_train_error)
# print(sumOf_sqrt_train_error/100)


#
# plt.plot(Y_train, color = 'red', label = 'Real data')
# plt.plot(prediction, color = 'blue', label = 'Predicted data')
# plt.title('Prediction')
# plt.legend()
# plt.show()
# correlation_matrix = np.corrcoef(X_train, Y_train)
# correlation_xy = correlation_matrix[0,1]
# r_squared = correlation_xy**2
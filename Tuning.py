import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# import sklearn
# scores = sorted(sklearn.metrics.SCORERS.keys())
# print(scores)

import warnings
warnings.filterwarnings("ignore")

# load dataset
dataframe = read_csv("Input/Train_Data.csv")
dataset = dataframe.values

X_train = dataset[:, 3:10]
Y_train = dataset[:, 2]

# ===== Standerdize Data set =====
# created scaler
scaler = StandardScaler()
# fit scaler on training dataset
scaler.fit(X_train)
# transform training dataset
X_train = scaler.transform(X_train)

# let's create a function that creates the model (required for KerasClassifier)
# while accepting the hyperparameters we want to tune
# we also pass some default values such as optimizer='rmsprop'
def ANN_model():
    # define model
    model = Sequential()
    model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)
# create the sklearn model for the network
model_batch_epoch_CV = KerasRegressor(build_fn=ANN_model, verbose=1)

# we choose the initializers that came at the top in our previous cross-validation!!
batches = [5,10,20,30]
epochs = [100,150,200,250,300]

# grid search for initializer, batch size and number of epochs
param_grid = dict(epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model_batch_epoch_CV,
                    param_grid=param_grid,
                    cv=10,
                    scoring='neg_mean_squared_error')
grid_result = grid.fit(X_train, Y_train)

# print results
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
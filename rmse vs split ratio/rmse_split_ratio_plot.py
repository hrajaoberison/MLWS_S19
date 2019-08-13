# Used to process image data
from PIL import Image

# Used to load in all files in a directory
import glob
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
import matplotlib.pyplot as plt

# will be used to open some of the csv data
import csv

# Will be used to solve a NN regression model
from sklearn.neural_network import MLPRegressor

# Used to split data and assess error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load in the images
rawData = []
files = sorted(glob.glob('/Users/hrajaoberison/Documents/LLE/2019-04-13 Learning Set/*.bmp'), key = os.path.getmtime)

for filename in files:
    img = Image.open(filename)
    rawData.append(np.array(img))
    img.close()
    
print(len(rawData))
print(rawData[0].shape)

# Put this into a 3125 x 64 x 64 matrix
data_preReshape = np.stack(rawData)

# Reshape into a 3125 x 4096 matrix for training
X = data_preReshape.reshape(data_preReshape.shape[0], -1)
X.shape


# Open the CSV file
yList = []
with open('/Users/hrajaoberison/Documents/LLE/2019-04-13 Learning Set/ZernikeCoeff_list.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        yList.append(row)
        
print(len(yList))
print(len(yList[0]))

# Put the Y values into a 3125 x 5 matrix
Y = np.stack(yList)
print(Y.shape)

# Normalize the features
# Normally we would subtract by the mean then divide by the standard deviation of the data to normalize,
# but since we're using an image as input we can take a shortcut and just divide by 255
X = X / 255

i = 0.02
rmse = []
split_ratio = []
while i<0.999:
    i = round(i,2)
    print(i)
    
    # Trains a NN with the given hyperparameters and dataset. X.shape = (m, n) and Y.shape = (m, k) where k is outputs.
    # returns an MLPRegressor, which you can find predictions from using mlp.predict(X)
    def trainNNRegression(X, Y, 
                          hidden_layer_sizes=(100,), 
                          activation='logistic', 
                          solver='lbfgs', 
                          alpha=0.0001, 
                          batch_size='auto', 
                          learning_rate= 'constant', 
                          learning_rate_init=0.001, 
                          power_t=0.5, 
                          max_iter=300, 
                          shuffle=True, 
                          random_state=None, 
                          tol=0.0001, 
                          verbose=False, 
                          warm_start=False, 
                          momentum=0.9, 
                          nesterovs_momentum=True, 
                          early_stopping=False, 
                          validation_fraction=0.1, 
                          beta_1=0.9, 
                          beta_2=0.999, 
                          epsilon=1e-08, 
                          n_iter_no_change=10):
        
        NNRegressor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change
        )
        NNRegressor.fit(X, Y)
    
        return NNRegressor
    
    # Split up our training data into a train set and a test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1-i, random_state=42)
    nnRegression = trainNNRegression(X_train, Y_train)
    Y_predict = nnRegression.predict(X_test)
    # Given a trained MLPRegressor, find the Root Mean Squared Error of the dataset
    rms_error = np.sqrt(mean_squared_error(Y_test, Y_predict))
    print(f"RMSE{[i]}: ", rms_error)
    rmse = np.hstack((rmse, rms_error))
    split_ratio = np.hstack((split_ratio, i))
    i = i+0.05
fig, ax = plt.subplots()
ax.plot(split_ratio, rmse)
ax.set_xlabel('Train_test')
ax.set_ylabel('Root Mean Squared Error')
ax.set_title('Train_test vs Root Mean Squared Error')
plt.show()
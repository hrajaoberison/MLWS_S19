import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from Image_Generation_script_edit1 import Image_Generation_script
import Image_Generation_script_edit1 as ZernikeReconstruct
import pandas
import pandas as pd
from PIL import Image
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Generate Zernike list
v_Zmax = np.array([3,3,2,1.5,1.5])
v_Zmax = np.reshape(v_Zmax, (1, v_Zmax.shape[0]))
v_rel = np.array([-1,-0.5,0,0.5,1])                             # Steps in each Zernike term, multiplied by the appropriate v_Zmax
v_rel = np.reshape(v_rel, (1, v_rel.shape[0]))
M_z = np.array([])
for i1 in range(0, v_rel.shape[1]):
   for i2 in range(0, v_rel.shape[1]):
       for i3 in range(0, v_rel.shape[1]):
           for i4 in range(0, v_rel.shape[1]):
               for i5 in range(0, v_rel.shape[1]):
                       a = np.array([v_rel[:,(i5)], v_rel[:,(i4)], v_rel[:,(i3)], v_rel[:,(i2)], v_rel[:,(i1)]])
                       a = np.reshape(a, (1, a.shape[0]))
                       b = v_Zmax*a
                       M_z = np.concatenate((M_z, b), axis=0) if M_z.size else b
       
# Randomize the order
v_ind = np.random.permutation(M_z.shape[0])
M_zrand = M_z[v_ind,:]

# Store Zernike list in csv file
df = pd.DataFrame(M_zrand)
filepath = '/Users/hrajaoberison/Documents/LLE/Testfolder/ZernikeCoeff_list.csv'
df.to_csv(filepath, index = False)

# Open the zernike coeff csv file
yList = []
with open('/Users/hrajaoberison/Documents/LLE/Testfolder/ZernikeCoeff_list.csv', 'r') as csvfile:
    y_reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    next(y_reader) # skip the first row
    for y_row in y_reader:
        yList.append(y_row)
# Put the Y values into a 3125 x 5 matrix, (Output or Target of NN)
Y = np.stack(yList)
print(Y.shape)

## Read and generate Zernike files

# Load Zernike terms
M_z = pd.read_csv(filepath)
M_z = np.matrix(M_z)

# Generate image data by varying Defocus distance(dz), in m
dz = np.linspace(1.8*10**-3, 2.5*10**-3, num = 4)
rmse = []
for dz_row in dz:    
    print(dz_row)
    NA = 0.1
    Lambda = 1.053*10**-6
    Nff_pts = 64
    Nnf_pts = 256
    
    # Set up far-field array
    rmax = 350*10**-6
    xfv = np.array([np.arange(-Nff_pts,(Nff_pts-1),2)/Nff_pts*rmax]) 
    yfv = xfv
    
    # Set up pupil array
    xv = np.array([np.arange(-Nnf_pts,(Nnf_pts-1),2)*1.1/Nnf_pts])
    yv = xv
    Xm, Ym = np.meshgrid(xv,yv)
    Rhom = np.sqrt(Xm**2+Ym**2)
    Phim = np.arctan2(Ym,Xm)
    Im = np.double(Rhom<=1.0)
    plt.figure(211)
    extent1 = np.min(xv), np.max(xv), np.min(yv), np.max(yv)
    im1 = plt.imshow(Im, extent=extent1, cmap ='jet')
    plt.colorbar(im1)
    
    # Test NF / FF
    Cv = np.array([0, 3, 3, -3, 2, 2])
    Cv = np.reshape(Cv, (1,Cv.shape[0]))
    args = (Xm, Ym, Cv)
    Wm = ZernikeReconstruct.ZernikeReconstruct(*args)
    plt.figure(212)
    extent3 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
    im2 = plt.imshow(Wm, extent=extent3, cmap ='jet')
    plt.colorbar(im2)
    
    arg1 = ((np.sqrt(Im)*np.exp(2j*np.pi*Wm)), xv*NA, yv*NA, xfv, yfv, Lambda, 1, dz_row)
    E_ff, x_ff, y_ff = Image_Generation_script(*arg1)
    
    I_ff = np.square(np.abs(E_ff))
    I_ff = I_ff/np.max(I_ff.flatten(order = 'F'))
    
    plt.figure(213)
    extent2 = np.min(xfv*10**6), np.max(xfv*10**6), np.min(yfv*10**6), np.max(yfv*10**6)
    im3 = plt.imshow(I_ff, extent=extent2, cmap ='gray')
    plt.colorbar(im3)
    j = Image.fromarray((I_ff*255).astype(np.uint8))
    j.save('TestImg1.bmp')
    plt.show()
    
    # Empty array to store the images
    rawData = []
    # dataSet = TemporaryFile()
    # Loop through and generate images
    for ii in tqdm(range(0, M_z.shape[0])):
        # Generate wavefront map for this iteration
        M_zy = M_z[ii,:]
        M_zy = np.append([0], M_zy)
        M_zy = np.reshape(M_zy, (1, M_zy.shape[0]))
        args3 = (Xm, Ym, M_zy)
        Wm = ZernikeReconstruct.ZernikeReconstruct(*args3)
        plt.figure(214)
        extent4 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
        im4 = plt.imshow(Wm, extent=extent4, cmap ='jet')
        plt.colorbar(im4)
    
        # Generate defocused far-field irradiance\
        args2 = (np.sqrt(Im)*np.exp(2j*np.pi*Wm), xv*NA, yv*NA, xfv, yfv, Lambda, 1, dz_row)
        E_ff, x_ff, y_ff = Image_Generation_script(*args2)
        I_ff = np.square(np.abs(E_ff))
        I_ff = I_ff/np.max(I_ff.flatten(order = 'F'))
    
        plt.figure(215)
        extent5 = np.min(xfv*10**6), np.max(xfv*10**6), np.min(yfv*10**6), np.max(yfv*10**6)
        im5 = plt.imshow(I_ff, extent=extent5, cmap ='gray')
        plt.colorbar(im5)
        dataPath = '/Users/hrajaoberison/Documents/LLE/Testfolder2/'
        suffix = '.bmp'
        k = Image.fromarray((I_ff*255).astype(np.uint8))
        k.save(dataPath + f"TrainingImg{ii}{suffix}")
        plt.show()
        
        # Load in the images
        img = Image.open(dataPath + f"TrainingImg{ii}{suffix}")
        rawData.append(np.array(img))
        img.close()
        os.unlink(dataPath + f"TrainingImg{ii}{suffix}")
    
    # Put this into a 3125 x 64 x 64 matrix
    data_preReshape = np.stack(rawData)
    
    # Reshape into a 3125 x 4096 matrix for training
    X = data_preReshape.reshape(data_preReshape.shape[0], -1)
    
    # Store Zernike list in csv file
    df = pd.DataFrame(X)
    filepath = '/Users/hrajaoberison/Documents/LLE/Testfolder2/ImgDataSet.csv'
    df.to_csv(filepath, index = False, header= False)

    # Open the image csv file
    xList = []
    with open('/Users/hrajaoberison/Documents/LLE/Testfolder2/ImgDataSet.csv', 'r') as csvfile:
        x_reader = csv.reader(csvfile, delimiter=',', quoting = csv.QUOTE_NONNUMERIC)
        for x_row in x_reader:
            xList.append(x_row)
    # Put the X values into a 3125 x 4096 matrix
    X = np.stack(xList)
    print(X.shape)
    os.unlink(filepath) # delete the current image dataset file

    # Normalize the features
    # Normally we would subtract by the mean then divide by the standard deviation of the data to normalize,
    # but since we're using an image as input we can take a shortcut and just divide by 255
    X = X / 255
    
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=42)
    nnRegression = trainNNRegression(X_train, Y_train)
    Y_predict = nnRegression.predict(X_test)
    # Given a trained MLPRegressor, find the Root Mean Squared Error of the dataset
    rms_error = np.sqrt(mean_squared_error(Y_test, Y_predict))
    print(f"RMSE{[dz_row]}: ", rms_error)
    rmse = np.hstack((rmse, rms_error))

fig, ax = plt.subplots()
ax.plot(dz, rmse)
ax.set_xlabel('Defocus distance [m]')
ax.set_ylabel('Root Mean Squared Error')
ax.set_title('Root Mean Squared Error Plot')
plt.show()
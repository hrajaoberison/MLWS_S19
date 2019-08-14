This folder contains two scripts:

ImageDataPreprocessing.py: Open images (.bmp) file and convert them to a 4D numpy array file of shape (N_img, height, width, channel). 
The whole point of this is to set the channel equal to 3, which works for convolution layers.

Zernike_Coeff_CNN_model.ipynb: Load the image data (4D numpy array file), preprocess, normalize and feed them to the CNN model.
It also predicts the Zernike Coefficients and determine the RMSE value.
After the training, it saves the architecture and weights of the trained model.

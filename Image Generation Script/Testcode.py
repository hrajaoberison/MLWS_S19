import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Image_Generation_script import Image_Generation_script
import Image_Generation_script as ZernikeReconstruct
import pandas
import pandas as pd
from PIL import Image

# Input variables
v_Zmax = np.array([3, 3, 2, 1.5, 1.5])
v_Zmax = np.reshape(v_Zmax, (1, v_Zmax.shape[0]))
dz = 2.266*10**-3
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

# Generate Zernike list
N_coeff = 5
N_img = round(4.7**N_coeff)
M_z = (np.ones((N_img,1))*v_Zmax)*(2*np.random.rand(N_img, N_coeff)-np.ones((N_img, N_coeff)))
       
# Randomize the order
v_ind = np.random.permutation(M_z.shape[0])
M_zrand = M_z[v_ind,:]

# Store Zernike list in csv file
df = pd.DataFrame(M_zrand)
filepath = '/Users/hrajaoberison/Documents/LLE/Testfolder2/ZernikeCoeff_list.csv'
df.to_csv(filepath, index = False, header= False)

### Read and generate Zernike files

## Load Zernike terms
M_z = pd.read_csv(filepath)
M_z = np.matrix(M_z)

# Loop through and generate images
for ii in tqdm(range(0, M_z.shape[0])):
    # Generate wavefront map for this iteration
    M_zy = M_z[ii,:]
    M_zy = np.append([0], M_zy)
    M_zy = np.reshape(M_zy, (1, M_zy.shape[0]))
    Wm = ZernikeReconstruct.ZernikeReconstruct(Xm, Ym, M_zy)
    plt.figure(214)
    extent4 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
    im4 = plt.imshow(Wm, extent=extent4, cmap ='jet')
    plt.colorbar(im4)

    # Generate defocused far-field irradiance
    args1 = (np.sqrt(Im)*np.exp(2j*np.pi*Wm), xv*NA, yv*NA, xfv, yfv, Lambda, 1, dz)
    E_ff, x_ff, y_ff = Image_Generation_script(*args1)
    I_ff = np.square(np.abs(E_ff))
    I_ff = I_ff/np.max(I_ff.flatten(order = 'F'))

    plt.figure(215)
    extent5 = np.min(xfv*10**6), np.max(xfv*10**6), np.min(yfv*10**6), np.max(yfv*10**6)
    im5 = plt.imshow(I_ff, extent=extent5, cmap ='gray')
    plt.colorbar(im5)
    save_results_to = '/Users/hrajaoberison/Documents/LLE/Testfolder2/'
    suffix = '.bmp'
    k = Image.fromarray((I_ff*255).astype(np.uint8))
    k.save(save_results_to + f"TrainingImg{ii}{suffix}")
    plt.show()
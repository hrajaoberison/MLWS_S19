import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Image_Generation_script import Image_Generation_script
import Image_Generation_script as ZernikeReconstruct
import pandas
import time
import pandas as pd
from PIL import Image

# Input variables
v_Zmax = np.array([3,3,2,1.5,1.5])
v_Zmax = np.reshape(v_Zmax, (1, v_Zmax.shape[0]))
v_rel = np.array([-1,-0.5,0,0.5,1])                         # Steps in each Zernike term, multiplied by the appropriate v_Zmax
v_Zmax = np.reshape(v_rel, (1, v_rel.shape[0]))
#dz = 1.53*10**-3
dz = 2.27*10**-3
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
ff = Im[1,:]
plt.figure(211)
extent1 = np.min(xv), np.max(xv), np.min(yv), np.max(yv)
im1 = plt.imshow(Im, extent=extent1, cmap ='jet')
plt.colorbar(im1)

# Test NF / FF
Bm = np.array([0, 3, 3, -3, 2, 2])
Bm = np.reshape(Bm, (1,Bm.shape[0]))
args = (Xm, Ym, Bm)
Wm = ZernikeReconstruct.ZernikeReconstruct(*args)
plt.figure(212)
extent3 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
im2 = plt.imshow(Wm, extent=extent3, cmap ='jet')
plt.colorbar(im2)

A = (np.sqrt(Im)*np.exp(2j*np.pi*Wm))
arg1 = ((np.sqrt(Im)*np.exp(2j*np.pi*Wm)), xv*NA, yv*NA, xfv, yfv, Lambda, 1, dz)
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

####

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
    args3 = (Xm, Ym, M_zy)
    Wm = ZernikeReconstruct.ZernikeReconstruct(*args3)
    plt.figure(214)
    extent4 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
    im4 = plt.imshow(Wm, extent=extent4, cmap ='jet')
    plt.colorbar(im4)

    # Generate defocused far-field irradiance
    args2 = (np.sqrt(Im)*np.exp(2j*np.pi*Wm), xv*NA, yv*NA, xfv, yfv, Lambda, 1, dz)
    E_ff, x_ff, y_ff = Image_Generation_script(*args2)
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
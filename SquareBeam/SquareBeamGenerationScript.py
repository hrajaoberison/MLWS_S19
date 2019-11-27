# Author: hrajaoberison
# Produces a sum of 2D Legendre polynomials weighted by the coefficients in 
# v_coeff.  Should replicate the behavior of LegendreMap, written by Seung-Whan Bahk.

# Import librabries
import numpy as np
import warnings
from scipy import special
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from PIL import Image

# Input variables
v_Zmax = np.array([[3, 3, 2, 1.5, 1.5]])
dz = 3.5*10**-3
f = 1.04
NA = 1
Lambda = 1.054*10**-6
Nff_pts = 512
Nnf_pts = 2048
v_region = np.array([[0,0,0.165,0.165]])

# Set up far-field array
rmax = 4.2*10**-6*Nff_pts/2
xfv = np.array([np.arange(-Nff_pts,(Nff_pts-1),2)/Nff_pts*rmax]) 
yfv = xfv

# Set up pupil array
xv = np.array([np.arange(-Nnf_pts,(Nnf_pts-1),2)*0.33/Nnf_pts])
yv = xv
Xm, Ym = np.meshgrid(xv,yv)
Im = np.double((np.abs(Xm)<=0.165)&(np.abs(Ym)<=0.165))

# Generate Zernike list
N_coeff = 5 # must be equal to max length of v_Zmax
if N_coeff != max(v_Zmax.shape):
   warnings.warn("Error: N_coeff must be equal to the length of v_Zmax array")
N_img = 10#round(4.7**N_coeff) # can be any integer
M_z = (np.ones((N_img,1))*v_Zmax)*(2*np.random.rand(N_img, N_coeff)-np.ones((N_img, N_coeff)))
       
# Randomize the order
v_ind = np.random.permutation(M_z.shape[0])
M_zrand = M_z[v_ind,:]

# Store Zernike list in csv file
df = pd.DataFrame(M_zrand)
filepath = '/Users/hrajaoberison/Documents/LLE/SquareBeam/Data/ZernikeCoeff_list.csv'
df.to_csv(filepath, index = False, header= False)

### Read and generate Zernike files

## Load Zernike terms
M_z = pd.read_csv(filepath,header=None)
M_z = np.matrix(M_z)

## Define Image generation function
def Image_Generation_script(En, x, y, xf, yf, wavelength, f, dz):
    Ny, Nx = np.shape(En)
    if x.shape[1] != Nx:
        warnings.warn("Error: x vector must be the same horizontal dimension of near field matrix")
    
    if x.shape[0] != 1:
        warnings.warn("Error: x vector must be a vector")
    
    if y.shape[1] != Ny:
        warnings.warn("Error: y vector must be the same vertical dimension of near field matrix")
    
    if y.shape[0] != 1:
        warnings.warn("Error: y vector must be a vector")

    if len(args1) < 5:
        wavelength = 1.053*10**-6

    if len(args1) < 6:
        f = 1

    if len(args1) < 7:
        dz = 0

    # Get the difference and mean between elements of vectors x and y respectively
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    if len(args1) < 3: 
        if Nx % 2:
            xf = (np.arange(-Nx+1,Nx-1,2))*wavelength*f/(2*Nx*dx)
        else:
            xf = (np.arange(-Nx,Nx-2,2))*wavelength*f/(2*Nx*dx)

    if len(args1) < 4: 
        if Ny % 2:
            yf = (np.arange(-Ny+1,Ny-1,2))*wavelength*f/(2*Ny*dy)
        else:
            yf = (np.arange(-Ny,Ny-2,2))*wavelength*f/(2*Ny*dy)
        
    # Force x and y to be row vectors
    x = np.reshape(x, (1,Nx))
    y = np.reshape(y, (1,Ny))

    # Centering x, y vector
    x = x - np.mean(x)
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x,y)
    rr = np.sqrt(xx**2 + yy**2)
    En = En*np.exp(-1j*np.pi/wavelength*dz/f**2*np.square(rr)) #  Adds defocus term to near-field

    dxf = np.mean(np.diff(xf))
    dyf = np.mean(np.diff(yf))

    kx = xf*2*np.pi/wavelength/f
    ky = yf*2*np.pi/wavelength/f

    A = np.exp(1j*np.matrix(ky).getH()*y) 
    B = np.exp(1j*np.matrix(x).getH()*kx)
    Ef = A*En*B*dx*dy/wavelength/f
    
    # Enforce Parseval's Theorem (energy conservation)
    NFenergy = np.sum(np.square(np.absolute(En.flatten('F'))))*dx*dy
    FFenergy = np.sum(np.square(np.absolute(Ef.flatten('F'))))*dxf*dyf
    Ef = Ef*np.sqrt(NFenergy/FFenergy)
    # Issue a warning if sampling is insufficient
    if np.max(np.absolute(xf)) > wavelength*f/dx/2 or np.max(np.absolute(yf)) > wavelength*f/dy/2:
        warnings.warn("Far-field crossed the replica boundary. Increase near-field sampling or reduce width of far-field grid.")
    return Ef, xf, yf

## Legendre 2D map function
def Legendre2DMap(v_coeff, Xm, Ym, v_region):
    if Xm.shape != Ym.shape:
        raise ValueError('Incompatible x and y matrices!')
        
    Xm_grid = (Xm - v_region[0,0])/v_region[0,2]
    Ym_grid = (Ym - v_region[0,1])/v_region[0,3]
    
    M_out = np.zeros(Xm.shape)
    
    j_ap = np.array(np.where(np.abs(Xm_grid[0,:])<=1))
    i_ap = np.array(np.where(np.abs(Ym_grid[:,0])<=1)).conj().transpose()
    
    # Loop through coefficients
    n=0
    for k in range(1, max(v_coeff.shape)+1):
        if k > np.sum(np.linspace(1,n+1,n+1))+0.1:
          n= n+1 
        j_k = (k-1)-np.sum(np.linspace(1,n,n))
        i_k = n - j_k
        vlx = special.lpmv(0, i_k, Xm_grid[0, j_ap])
        vlx = np.squeeze(vlx[0,:]).reshape((1,max(vlx.shape)))

        vly = special.lpmv(0, j_k, Ym_grid[i_ap, 1].conj().transpose())
        vly = np.squeeze(vly[0,:]).reshape((1,max(vly.shape)))
        
        Pm_k = (vly.conj().transpose())*vlx
        M_out[i_ap,j_ap] = M_out[i_ap,j_ap] + v_coeff[0, k-1]*Pm_k
    return M_out

if __name__ == "__main__":
    # Loop through and generate images
    for ii in tqdm(range(0, M_z.shape[0])):
        # Generate wavefront map for this iteration
        M_zy = M_z[ii,:]
        M_zy = np.append([0], M_zy)
        M_zy = np.reshape(M_zy, (1, M_zy.shape[0]))
        Wm = Legendre2DMap(M_zy, Xm, Ym, v_region)

        plt.figure(214)
        extent4 = np.min(xv*NA), np.max(xv*NA), np.min(yv*NA), np.max(yv*NA)
        im4 = plt.imshow(Wm, extent=extent4, cmap ='jet')
        plt.colorbar(im4)
    
        # Generate defocused far-field irradiance
        args1 = (np.sqrt(Im)*np.exp(2j*np.pi*Wm), xv*NA, yv*NA, xfv, yfv, Lambda, f, dz)
        E_ff, x_ff, y_ff = Image_Generation_script(*args1)
        I_ff = np.square(np.abs(E_ff))
        I_ff = I_ff/np.max(I_ff.flatten(order = 'F'))
    
        plt.figure(215)
        extent5 = np.min(xfv*10**6), np.max(xfv*10**6), np.min(yfv*10**6), np.max(yfv*10**6)
        im5 = plt.imshow(I_ff, extent=extent5, cmap ='gray')
        plt.colorbar(im5)
        save_results_to = '/Users/hrajaoberison/Documents/LLE/SquareBeam/Data/' # where the images are saved
        suffix = '.bmp'
        k = Image.fromarray((I_ff*255).astype(np.uint8))
        k.save(save_results_to + f"TrainingImg{ii}{suffix}")
        plt.show()
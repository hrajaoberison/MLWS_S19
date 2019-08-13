###### Written by Heri Rajaoberison ######
import sys
import numpy as np
import warnings
import matplotlib.pyplot as plt
import GeneratesArrays
  
def PropagateNF2FF_DFT(En, x, y, xf, yf, wavelength, f, dz):
    Ny, Nx = np.shape(En)
    
    if x.shape[1] != Nx:
        warnings.warn("Error: x vector must be the same horizontal dimension of near field matrix")
    
    if x.shape[0] != 1:
        warnings.warn("Error: x vector must be a vector")
    
    if y.shape[1] != Ny:
        warnings.warn("Error: y vector must be the same vertical dimension of near field matrix")
    
    if y.shape[0] != 1:
        warnings.warn("Error: y vector must be a vector")
        
    if len(GeneratesArrays.args) < 5:
        wavelength = 1.053*10**-6

    if len(GeneratesArrays.args) < 6:
        f = 1
    
    if len(GeneratesArrays.args) < 7:
        dz = 0
    
    # Get the difference and mean between elements of vectors x and y respectively
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    
    if len(GeneratesArrays.args) < 3: 
        if Nx % 2:
            xf = (np.arange(-Nx+1,Nx-1,2))*wavelength*f/(2*Nx*dx)
        else:
            xf = (np.arange(-Nx,Nx-2,2))*wavelength*f/(2*Nx*dx)
    
    if len(GeneratesArrays.args) < 4: 
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

if __name__ == "__main__":
    PropagateNF2FF_DFT(*GeneratesArrays.args)
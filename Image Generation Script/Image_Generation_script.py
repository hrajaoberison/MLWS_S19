# Script for generatiing far-fields for machine learning sets
import numpy as np
import warnings
import Testcode

######
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

    if len(Testcode.args1) < 5:
        wavelength = 1.053*10**-6

    if len(Testcode.args1) < 6:
        f = 1

    if len(Testcode.args1) < 7:
        dz = 0

    # Get the difference and mean between elements of vectors x and y respectively
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    if len(Testcode.args1) < 3: 
        if Nx % 2:
            xf = (np.arange(-Nx+1,Nx-1,2))*wavelength*f/(2*Nx*dx)
        else:
            xf = (np.arange(-Nx,Nx-2,2))*wavelength*f/(2*Nx*dx)

    if len(Testcode.args1) < 4: 
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

def ZernikeReconstruct(Xm, Ym, Zv):
    # Make sure Zv is a column vector
    if Zv.shape[1] != 1 or Zv.shape[0] < Zv.shape[1]:
        Zv= np.reshape(Zv, (Zv.shape[1],1))

    # Define pupil (all real-valued Wm within unit circle)
    Rhom = np.sqrt(Xm**2+Ym**2)
    Thetam = np.arctan2(Ym,Xm)
    
    Mp = Rhom<=1
    Wm = np.zeros(Xm.shape)
    
    Thetav = np.array([Thetam.flatten('F')[Mp.flatten('F')]])
    Rhov = np.array([Rhom.flatten('F')[Mp.flatten('F')]])
    
    # Define Zernike polynomial terms with 0-15 rows
    Pm = np.empty((Thetav.shape[1],Thetav.shape[1]))
    Pm[0:1] = np.ones(Thetav.shape)  # Index 1 column of Z0
    Pm[1:2] = Rhov*np.cos(Thetav)  # Z1, x tilt
    Pm[2:3] = Rhov*np.sin(Thetav)  # Z2, y tilt
    Pm[3:4] = 2*np.square(Rhov)-1          # Z3, defocus
    Pm[4:5] = np.square(Rhov)*np.cos(2*Thetav) # Z4, 45-deg astig.
    Pm[5:6] = np.square(Rhov)*np.sin(2*Thetav)    # Z5, 45-deg astig.
    Pm[6:7] = (3*Rhov**3-2*Rhov)*np.cos(Thetav)    # Z6, x coma
    Pm[7:8] = (3*Rhov**3-2*Rhov)*np.sin(Thetav)    # Z7, y coma
    Pm[8:9] = 6*Rhov**4-6*Rhov**2+1    # Z8, spherical
    Pm[9:10] = Rhov**3*np.cos(3*Thetav)    # Z9
    Pm[10:11] = Rhov**3*np.sin(3*Thetav)    # Z10
    Pm[11:12] = (4*Rhov**4-3*Rhov**2)*np.cos(2*Thetav)  # Z11
    Pm[12:13] = (4*Rhov**4-3*Rhov**2)*np.sin(2*Thetav)  # Z12
    Pm[13:14] = (10*Rhov**5-12*Rhov**3+3*Rhov)*np.cos(Thetav)  # Z13
    Pm[14:15] = (10*Rhov**5-12*Rhov**3+3*Rhov)*np.sin(Thetav)  # Z14
    Pm[15:16] = 20*Rhov**6-30*Rhov**4+12*Rhov**2-1    # Z15
    Pm = np.transpose(Pm)
    if Zv.shape[0]<16:
        Pm = Pm[:,:Zv.shape[0]] # delete columns of Pm starting from Zv.shape[0]+1  
    Zm = np.matmul(Pm, Zv) # a column vector = multiple rows same as Pm rows.
    Wm[Rhom<=1] = np.sum(Zm, axis = 1)
    Wm = np.rot90(Wm)
    return Wm
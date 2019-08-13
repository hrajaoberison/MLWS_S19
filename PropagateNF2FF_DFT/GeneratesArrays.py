####### Written by Heri Rajaoberison ######
#  Generates arrays to be fed into PropagateNF2FF_DFT.m
import sys
import numpy as np
import warnings
import matplotlib.pyplot as plt
from PropagateNF2FF_DFT import PropagateNF2FF_DFT

Nff_pts = 64
Nnf_pts = 256
NA = 0.1
Lambda = 1.053*10**-6                   # wavelength in meters
EFL = 1                                 # Focal length, in meters
dz = 1.5*10**-3                         # Defocus distance, in meters

# Set up far-field array
rmax = 350*10**-6
xfv = np.array([np.arange(-Nff_pts,(Nff_pts-1),2)/Nff_pts*rmax])         # Create a row vector xfv
yfv = xfv

# Set up far-field array
xv = np.array([np.arange(-Nnf_pts,(Nnf_pts-1),2)*1.1/Nnf_pts])
yv = xv

# Set up pupil array
Xm, Ym = np.meshgrid(xv,yv)
Rhom = np.sqrt(Xm**2+Ym**2)
Phim = np.arctan2(Ym,Xm)
Im = np.double(Rhom<=1.0)

plt.figure(211)
extent1 = np.min(xv), np.max(xv), np.min(yv), np.max(yv)
im1 = plt.imshow(Im, extent=extent1, cmap ='jet')
plt.colorbar(im1)
  
args = (np.sqrt(Im), xv*NA, yv*NA, xfv, yfv, Lambda, EFL, dz)
E_ff, x_ff, y_ff = PropagateNF2FF_DFT(*args)
I_ff = np.square(np.abs(E_ff))
I_ff = I_ff/np.max(I_ff.flatten(order = 'F'))

plt.figure(212)
extent2 = np.min(xfv*10**6), np.max(xfv*10**6), np.min(yfv*10**6), np.max(yfv*10**6)
im2 = plt.imshow(I_ff, extent=extent2, cmap ='gray')
plt.colorbar(im2)
plt.show()
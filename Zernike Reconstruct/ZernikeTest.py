# Script to test ZernikeReconstruct
import numpy as np
import matplotlib.pyplot as plt
import ZernikeReconstruct as ZernikeReconstruct

xv = np.array([np.arange(-1.10, 1.11, 0.05)])


yv = xv
Xm, Ym = np.meshgrid(xv, yv)
Cv = np.array([np.arange(1,7,1)])

args = (Xm, Ym, Cv)
Wm = ZernikeReconstruct.ZernikeReconstruct(*args)

plt.figure(213)
extent3 = np.min(xv), np.max(xv), np.min(yv), np.max(yv)
im1 = plt.imshow(Wm, extent=extent3, cmap ='jet')
plt.colorbar(im1)
plt.show()


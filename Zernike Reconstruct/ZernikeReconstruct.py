import numpy as np
import ZernikeTest

def ZernikeReconstruct(Xm, Ym, Zv):
    # Make sure Zv is a column vector
    if Zv.shape[1] != 1 or Zv.shape[0] < Zv.shape[1]:
        Zv= np.reshape(Zv, (Zv.shape[1],1))
    print("Zv=", Zv.shape)

    # Define pupil (all real-valued Wm within unit circle)
    Rhom = np.sqrt(Xm**2+Ym**2)
    Thetam = np.arctan2(Ym,Xm)
    print("Thetam=", np.max(Thetam))
    print("Rhom=", np.min(Rhom))
    
    Mp = Rhom<=1
    Mp = Mp.astype(np.int)
    print("Mp=", Mp.shape)
    Wm = np.zeros(Xm.shape)
    print("Wm=", Wm.shape)
    
    Rhov = np.array([Rhom[Rhom<=1].flatten('F')])
    Thetav = np.array([Thetam[Rhom<=1].flatten('F')])
    print("Rhov=", Rhov.shape)
    print("Thetav=", Thetav.shape)
    
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
    print("Zv=", Zv.shape)
    print("Pm=", Pm.shape)
    Zm = np.matmul(Pm, Zv) # a column vector = multiple rows same as Pm rows.
    print("Zm=", Zm)
    Wm[Rhom<=1] = np.sum(Zm, axis = 1)
    print("Wm=", Wm)
    return Wm

if __name__ == "__main__":  
    Wm = ZernikeReconstruct(*ZernikeTest.args)
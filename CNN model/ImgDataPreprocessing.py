# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 08:57:37 2019

@author: hraj
"""

import numpy as np
import os
import glob
from PIL import Image

rawData=[]

path = r'C:\Users\hraj\Downloads\ImgData_14(3000x64x64)\*.bmp'
files = sorted(glob.glob(path), key=os.path.getmtime)

for filename in files:
    img = Image.open(filename).convert('RGB')
    rawData.append(np.array(img))
    img.close()
    
X = np.stack(rawData)

Img_file_name = 'ImgData_coeff_14(3000x64x64)'
np.save(Img_file_name, X)
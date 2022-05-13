import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy import ndimage, misc

import matplotlib.pyplot as plt

from PIL import Image

#gabor filter is usually used in texture analysis, edge detection, feature extraction, disparity estimation (


def deginrad(degree):
    

    radiant = 2*np.pi/360 * degree
    return radiant




colore=str(input("inserisci colore"))

if colore == " b":


   

    theta = deginrad(25)   # unit circle: left: -90 deg, right: 90 deg, straight: 0 deg

    path = '/home/oli/robot-code/color-detector-blue/blue/1.jpg'



    g_kernel = cv2.getGaborKernel((4, 4), 20, theta, 5, 0, 0, ktype=cv2.CV_32F)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)

    cv2.imwrite("reference_gabor.png", filtered_img)
    cv2.imshow("reference_gabor.png", filtered_img)
    cv2.waitKey(0)

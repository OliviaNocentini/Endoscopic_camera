import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import os

path = '/home/oli/robot-code/color-detector/'

for index in range (1,80):

    path1= os.path.join(path, ''.join([str(index), '.jpg']))
    
    image = cv2.imread(path1)                
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define blue color range
    light_blue = np.array([30,30,30])
    dark_blue = np.array([255,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, light_blue, dark_blue)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(image,image, mask= mask)
        
    cv2.imshow("original image and Color Detected", np.hstack((image,output)))
    cv2.imshow("mask", mask)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    image1 = cv2.imread("/home/oli/robot-code/color-detector/24.jpg")                
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    # define blue color range
    light_blue = np.array([110,50,50])
    dark_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, light_blue, dark_blue)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(image1,image, mask= mask)


    cv2.imshow("mask", mask)    
    cv2.imshow("original image and Color Detected", np.hstack((image,output)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #print("mask", mask)
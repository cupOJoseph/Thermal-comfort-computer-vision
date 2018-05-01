# python2.7

# image processing
import cv2
import numpy as np


color = "data/2018.01.22-19.41.50_01" # color
thermal = "data/2018.01.22-19.41.50_11" #thermal


kWidthIR = 160
kHeightIR = 120
def loadIR(filename):
	array = np.fromfile(filename, dtype=np.uint16)
	array = array.reshape((kHeightIR, kWidthIR))

	array = cv2.normalize(array,0.0,1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
	return cv2.flip(array, -1)

kWidthRGB = 1280
kHeightRGB = 720
kChannelsRGB = 3
def loadRGB(filename):
	array = np.fromfile(filename, dtype=np.uint8)
	array = array.reshape((kHeightRGB, kWidthRGB, kChannelsRGB))
	return array


colorImg = loadRGB(color)
thermalImg = loadIR(thermal)


cv2.imshow('color',colorImg)
cv2.imshow('thermal',thermalImg)
cv2.waitKey(0) #wait to press any key
cv2.destroyAllWindows()
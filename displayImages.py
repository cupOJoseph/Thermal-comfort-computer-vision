# python2.7

# image processing
import cv2
import numpy as np


color = "data/2018.01.22-19.41.50_01" # color
thermal = "data/2018.01.22-19.41.50_11" #thermal


kWidthIR = 160
kHeightIR = 120
def loadIR(filename):
	array = np.fromfile(filename, dtype=np.uint16)#try 8, messes with size of image
	array = array.reshape((kHeightIR, kWidthIR))

	array = cv2.normalize(array,0.0,1.0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)#try CV_8U
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

#TODO Joseph Code
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 20
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 16
params.maxArea = 500

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.05

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

params.filterByColor = False

print params.filterByColor
print params.filterByArea
print params.filterByCircularity
print params.filterByInertia
print params.filterByConvexity

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
im = cv2.imread("data/karl_ir_crop.png", cv2.IMREAD_GRAYSCALE)
keypoints = detector.detect(im)

#===================== display

cv2.imshow('color',colorImg)
cv2.imshow('thermal',thermalImg)
cv2.waitKey(0) #wait to press any key
#cv2.destroyAllWindows()

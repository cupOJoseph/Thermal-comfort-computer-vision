#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread("karl_ir_crop.png", cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 255


# Filter by Area.
params.filterByArea = True
params.minArea = 3
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
keypoints = detector.detect(im)

# Draw detected blobs as red circles..
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
reg_im = cv2.imread("regface.png")
reg_h, reg_w, regd = reg_im.shape
ir_h, ir_w = im.shape

h_ratio = reg_h / ir_h
w_ratio = reg_w / reg_h

keypoints_reg = keypoints

for idx, point in enumerate(keypoints):
    print(idx, point)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#translated_im = cv2.drawKeypoints(reg_im, keypoints_reg, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.imshow("Translated", translated_im)
cv2.waitKey(0)

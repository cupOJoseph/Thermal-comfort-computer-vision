import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def loadIR(filename):
	kWidthIR = 160
	kHeightIR = 120
	array = np.fromfile(filename, dtype=np.uint16)
	array = array.reshape((kHeightIR, kWidthIR))
	array = cv2.normalize(array,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
	return cv2.flip(array, -1)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the IR image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
thermal = "../data/2018.01.22-19.41.50_11" #thermal
frame = loadIR(thermal)
gray = frame.copy()
MINSIZE = (40, 40)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Face detection - get small region
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cascPath = "haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


faces = []
index = np.arange(0.0, 0.7, 0.1)
for i in index:
	# Detect faces in the image
	faceList = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1+i,
		minNeighbors=0,
		minSize=MINSIZE,
		flags = cv2.CASCADE_SCALE_IMAGE
	)
	for f in faceList:
		faces.append(f)

n = len(faces)
print("Found {0} faces!".format(n))
if n == 0:
	print "--No faces detected. Quitting."
	sys.exit(0)



# Average results
x = 0
y = 0
w = 0
h = 0
for (x_, y_, w_, h_) in faces:
	x = x + x_
	y = y + y_
	w = w + w_
	h = h + h_
x = int(x / n)
y = int(y / n)
w = int(w / n)
h = int(h / n)

# if imgType == 'ir':
# 	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

# Draw a rectangle around the faces
# cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),1) #original rectangle
# cv2.rectangle(frame, (x-int(w*0.1), y-int(h*0.2)), (x+int(w*1.1), y+int(h*1.2)), (0, 255, 0), 1) # larger
# cv2.rectangle(frame, (x+int(w*0.15), y+int(h*0.15)), (x+int(w*0.85), y+int(h*0.85)), (0, 255, 255), 1) # smaller 

# cv2.imshow("Face found", frame)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract the skin region of the face
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x1 = x+int(w*0.15)
x2 = x+int(w*0.85)
y1 = y+int(h*0.15)
y2 = y+int(h*0.85)
skinFrame = frame[y1:y2, x1:x2]
# cv2.imshow("skin frame",skinFrame) #regular skin frame

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Crop the head frame out
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x1 = x-int(w*0.1)
x2 = x+int(w*1.1)
y1 = y-int(h*0.2)
y2 = y+int(h*1.2)
headFrame = frame[y1:y2, x1:x2]
# cv2.imshow("head frame",headFrame)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find mask for skin color from the skin frame
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
height, width = frame.shape[:2]
frame = cv2.resize(frame, (width/8, height/8), interpolation = cv2.INTER_AREA)
# cv2.imshow("skinFrame",skinFrame) #hsv skin frame


# h = np.zeros(64,dtype='int')
# s = np.zeros(64,dtype='int')
# v = np.zeros(64,dtype='int')

# for k,c in enumerate((h,s,v)):
# 	for i in range(0,frame.shape[0]):
# 		for j in range(0,frame.shape[1]):
# 			for x in range(0,64):
# 				if frame[i,j,k] < x*4:
# 					c[x-1] = c[x-1] + 1
# 					break

# print h
# print s
# print v

# x = np.arange(0,255,4)


# plt.subplot(221)
# plt.bar(x,h)
# plt.ylabel('hue')

# plt.subplot(222)
# plt.bar(x,s)
# plt.ylabel('saturation')

# plt.subplot(223)
# plt.bar(x,v)
# plt.ylabel('value')
# plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply the skin color mask on the head frame
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# headFrame = cv2.GaussianBlur(headFrame,(25,25),0)
# cv2.medianBlur(headFrame,35)
# cv2.blur(headFrame,(25,25))

# lower = np.array([0.0, 0.2, 0.1], dtype = "float32")
# upper = np.array([0.1, 0.7, 1.0], dtype = "float32")

lower = np.array([0, 25, 25], dtype = "uint8")
upper = np.array([40, 190, 255], dtype = "uint8")

lower = np.array([0, 25, 25], dtype = "uint8")
upper = np.array([25, 175, 255], dtype = "uint8")

# lower = np.array([16, 16, 15], dtype = "uint8")
# upper = np.array([70, 75, 255], dtype = "uint8")

# lower = np.array([0, 0, 0], dtype = "uint8")
# upper = np.array([80, 275, 255], dtype = "uint8")

#rgb
# lower = np.array([0, 10, 15], dtype = "uint8")
# upper = np.array([80, 75, 75], dtype = "uint8")

# #rgb
# lower = np.array([10, 20, 15], dtype = "uint8")
# upper = np.array([80, 75, 75], dtype = "uint8")


# apply the mask to the frame
skinMask = cv2.inRange(headFrame, 127, 255)
# cv2.imshow("skinMask",skinMask)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Erode the skin mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
# cv2.imshow("eroded",skinMask)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find the largest connected component of the skin mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#source: https://www.programcreek.com/python/example/89340/cv2.connectedComponentsWithStats
def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
						smooth_boundary=False, kernel_size=11):
		'''Select the largest object from a binary image and optionally
		fill holes inside it and smooth its boundary.
		Args:
			img_bin (2D array): 2D numpy array of binary image.
			lab_val ([int]): integer value used for the label of the largest 
					object. Default is 255.
			fill_holes ([boolean]): whether fill the holes inside the largest 
					object or not. Default is false.
			smooth_boundary ([boolean]): whether smooth the boundary of the 
					largest object using morphological opening or not. Default 
					is false.
			kernel_size ([int]): the size of the kernel used for morphological 
					operation. Default is 15.
		Returns:
			a binary image as a mask for the largest object.
		'''
		n_labels, img_labeled, lab_stats, _ = \
			cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
											 ltype=cv2.CV_32S)
		largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
		largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
		largest_mask[img_labeled == largest_obj_lab] = lab_val
		# import pdb; pdb.set_trace()
		if fill_holes:
			bkg_locs = np.where(img_labeled == 0)
			bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
			img_floodfill = largest_mask.copy()
			h_, w_ = largest_mask.shape
			mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
			cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
						  newVal=lab_val)
			holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
			largest_mask = largest_mask + holes_mask
		if smooth_boundary:
			kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
			largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, 
											kernel_)
			
		return largest_mask 


largestObjectMask = select_largest_obj(skinMask)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dilate the new skin mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
largestObjectMask = cv2.dilate(largestObjectMask, kernel, iterations = 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find contour points of the largestObjectMask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img, contours, hierarchy = cv2.findContours(largestObjectMask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cImg = cv2.cvtColor(largestObjectMask, cv2.COLOR_GRAY2BGR)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fit an ellipse to the contour
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ellipse = cv2.fitEllipse(contours[0])
# (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
# cv2.imshow("contours",cImg)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a mask that is the shape of the ellipse
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
eMask = np.zeros(largestObjectMask.shape)
cv2.ellipse(eMask,ellipse, 255,-1)
eMask = cv2.inRange(eMask, 127, 255) #convert to binary


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply the mask to the head Frame
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ellipseHead = cv2.bitwise_and(headFrame, headFrame, mask = eMask)
# ellipseHead = cv2.cvtColor(ellipseHead, cv2.COLOR_HSV2BGR)
cv2.imshow("ellipseHeadIR",ellipseHead)

cv2.imwrite("karlHeadIR.png",ellipseHead)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cv2.waitKey(0)
cv2.destroyAllWindows()



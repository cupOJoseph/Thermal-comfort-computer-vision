import numpy as np
import cv2
import sys


# def auto_canny(image, sigma=0.33):
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged # return the edged image

def loadBGR(filename):
	kWidthRGB = 1280
	kHeightRGB = 720
	kChannelsRGB = 3
	array = np.fromfile(filename, dtype=np.uint8)
	array = array.reshape((kHeightRGB, kWidthRGB, kChannelsRGB))
	return array

color1 = "../karl_binaryAll/2018.01.22-19.41.50_01" #karl
color2 = "../karl_binaryAll/2018.01.22-19.41.50_10" #karl
# color1 = '../small_sets/2018.01.22-13.26.35_01' # ab
# color2 = '../small_sets/2018.01.22-13.26.35_10' # ab

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load the binary files as bgr
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img1 = loadBGR(color1)
img2 = loadBGR(color2)
MINSIZE = (200, 200)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Face detection (img1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath) # Create the haar cascade
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

faces = []
index = np.arange(0.0, 0.7, 0.1)
for i in index:
	# Detect faces in the image
	faceList = faceCascade.detectMultiScale(
		gray1,
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
x = 0; y = 0; w = 0; h = 0
for (x_, y_, w_, h_) in faces:
	x = x + x_
	y = y + y_
	w = w + w_
	h = h + h_
x = int(x / n)
y = int(y / n)
w = int(w / n)
h = int(h / n)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Crop the head frames out
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x1 = x-int(w*0.1)
x2 = x+int(w*1.1)
y1 = y-int(h*0.3)
y2 = y+int(h*1.3)
img1 = img1[y1:y2, x1:x2, :]
img2 = img2[y1:y2, x1:x2, :]
# cv2.imshow("head frame 1",img1)
headFrame1 = img1
headFrame2 = img2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Auto_canny
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img1 = auto_canny(img1)
img2 = auto_canny(img2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find the difference between the two canny images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
diff = abs(img1 - img2)
cv2.imshow("canny difference",diff)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dilate the difference
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
diff = cv2.dilate(diff, kernel, iterations = 8)
# cv2.imshow("dilated",diff)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Erode the skin mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
diff = cv2.erode(diff, kernel, iterations = 3)
# cv2.imshow("eroded",diff)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find the largest connected component of the skin mask
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#source: https://www.programcreek.com/python/example/89340/cv2.connectedComponentsWithStats
def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
						smooth_boundary=True, kernel_size=3):
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
			cv2.connectedComponentsWithStats(img_bin, connectivity=8,ltype=cv2.CV_32S)
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
			cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
			holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
			largest_mask = largest_mask + holes_mask
		if smooth_boundary:
			kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
			largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)
			
		return largest_mask 


diff = select_largest_obj(diff)
# cv2.imshow("largest Object",diff)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Erode the largest object
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
diff = cv2.erode(diff, kernel, iterations = 2)
# cv2.imshow("eroded largest",diff)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dilate the difference
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
diff = cv2.dilate(diff, kernel, iterations = 5)
# cv2.imshow("dilated",diff)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Find contour points of the largest object
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
img, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cImg = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
try:
	cv2.drawContours(cImg, contours, -1, (0,255,0), 2,maxLevel=0)
except:
	print "cannot find contours. need to dilate more"
	sys.exit(0)
# cv2.imshow("diff contours",cImg)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fit an ellipse to the contour
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ellipse = cv2.fitEllipse(contours[0])
# (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
cv2.ellipse(cImg, ellipse, (255,255,0), 2)
cv2.imshow("contours & ellipse",cImg)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a mask that is the shape of the ellipse
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
eMask = np.zeros(diff.shape)
cv2.ellipse(eMask,ellipse, 255,-1)
eMask = cv2.inRange(eMask, 127, 255) #convert to binary


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Apply the mask to the head Frame
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ellipseHead = cv2.bitwise_and(headFrame1, headFrame1, mask = eMask)
# ellipseHead = cv2.cvtColor(ellipseHead, cv2.COLOR_HSV2BGR)
cv2.imshow("ellipseHead",ellipseHead)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cv2.imwrite("../results/karlHeadColor.png",ellipseHead)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cv2.waitKey(0)
cv2.destroyAllWindows()
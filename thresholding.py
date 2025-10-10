import cv2 as cv
 
img=cv.imread('cat.jpeg')
cv.imshow('original image',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray image",gray)

#simple thresholding
threshold,thresh=cv.threshold(gray,99,255,cv.THRESH_BINARY)
cv.imshow('Simple Threshold',thresh)


#simple thresholding INVERSE
threshold,thresh_inv=cv.threshold(gray,99,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple Threshold_inverse',thresh_inv)


#Adaptive Thresholding
adp_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,17,1)
cv.imshow('adaptive Thresholding',adp_thresh)


cv.waitKey(0)
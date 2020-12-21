import cv2
import numpy as np
import matplotlib.pyplot as plt
import pysift
from PIL import Image


orb = cv2.ORB.create()
print('finder type: {}'.format(orb))
img = cv2.imread('./inputs/1.jpg')
img2 = cv2.imread('./inputs/2.jpg')
img3 = cv2.imread('./inputs/3.jpg')
print(img.shape)
img_feat = cv2.detail.computeImageFeatures2(orb, img)
img_feat2 = cv2.detail.computeImageFeatures2(orb, img2)
img_feat3 = cv2.detail.computeImageFeatures2(orb, img3)
kp = img_feat.getKeypoints()
print((kp))
desc = img_feat.descriptors
desc = cv2.UMat.get(desc)
print(desc)
img_ = img.copy()
img_ = cv2.drawKeypoints(img,kp,outImage = img_,color=(0,255,0), flags=0)
cv2.imshow('keypoints map', img_)
cv2.waitKey(0)
cv2.imwrite('./visualization/1.jpg', img_)

img_2 = img2.copy()
img_2 = cv2.drawKeypoints(img2,kp,outImage = img_2,color=(0,255,0), flags=0)
cv2.imshow('keypoints map', img_2)
cv2.waitKey(0)
cv2.imwrite('./visualization/2.jpg', img_2)

img_3 = img.copy()
img_3 = cv2.drawKeypoints(img3,kp,outImage = img_3,color=(0,255,0), flags=0)
cv2.imshow('keypoints map', img_3)
cv2.waitKey(0)
cv2.imwrite('./visualization/3.jpg', img_3)

# kp1, des1 = orb.detectAndCompute(img,None)
# print(des1)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pysift
from PIL import Image


orb = cv2.ORB.create()
print('finder type: {}'.format(orb))
img = cv2.imread('./inputs/1.jpg')
print(img.shape)
img_feat = cv2.detail.computeImageFeatures2(orb, img)
kp = img_feat.getKeypoints()
print(len(kp))
img2 = img.copy()
img2 = cv2.drawKeypoints(img,kp,outImage = img2,color=(0,255,0), flags=0)
cv2.imshow('keypoints map', img2)
cv2.waitKey(0)
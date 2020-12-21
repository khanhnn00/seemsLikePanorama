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
# print(img.shape)
img_feat = cv2.detail.computeImageFeatures2(orb, img)
img_feat2 = cv2.detail.computeImageFeatures2(orb, img2)
img_feat3 = cv2.detail.computeImageFeatures2(orb, img3)
features_12 = []
features_23 = []
features_13 = []
features = []
features_12.append(img_feat)
features_12.append(img_feat2)
features_23.append(img_feat2)
features_23.append(img_feat3)
features_13.append(img_feat)
features_13.append(img_feat3)
features.append(img_feat)
features.append(img_feat2)
features.append(img_feat3)
kp = img_feat.getKeypoints()
kp2 = img_feat2.getKeypoints()
kp3 = img_feat3.getKeypoints()
# print((kp))
desc = img_feat.descriptors
desc = cv2.UMat.get(desc)
# print(desc)
# img_ = img.copy()
# img_ = cv2.drawKeypoints(img,kp,outImage = img_,color=(0,255,0), flags=0)
# cv2.imshow('keypoints map', img_)
# cv2.waitKey(0)
# cv2.imwrite('./visualization/1.jpg', img_)

# img_2 = img2.copy()
# img_2 = cv2.drawKeypoints(img2,kp,outImage = img_2,color=(0,255,0), flags=0)
# cv2.imshow('keypoints map', img_2)
# cv2.waitKey(0)
# cv2.imwrite('./visualization/2.jpg', img_2)

# img_3 = img.copy()
# img_3 = cv2.drawKeypoints(img3,kp,outImage = img_3,color=(0,255,0), flags=0)
# cv2.imshow('keypoints map', img_3)
# cv2.waitKey(0)
# cv2.imwrite('./visualization/3.jpg', img_3)
matcher = cv2.detail_BestOf2NearestMatcher()
p12 = matcher.apply2(features_12)[1:2]
p23 = matcher.apply2(features_23)[1:2]
p13 = matcher.apply2(features_13)[1:2]
p = matcher.apply2(features)
counts = 4
# print(type(p12))



mask12 = p12[0].getMatches()

img3_ = img.copy()

img3_ = cv2.drawMatches(img,kp,img2,kp2,mask12,matchColor=(0,255,0), singlePointColor=(255,0,0), outImg = img3_, flags=2)
cv2.imshow('keypoints map', img3_)
cv2.waitKey(0)
cv2.imwrite('./visualization/{}.jpg'.format(counts), img3_)
counts = counts + 1

mask23 = p23[0].getMatches()

img3_ = img2.copy()
img3_ = cv2.drawMatches(img2,kp2,img3,kp3,mask23,matchColor=(0,255,0), singlePointColor=(255,0,0), outImg = img3_, flags=2)
cv2.imshow('keypoints map', img3_)
cv2.waitKey(0)
cv2.imwrite('./visualization/{}.jpg'.format(counts), img3_)
counts = counts + 1

mask13 = p13[0].getMatches()

img3_ = img.copy()
img3_ = cv2.drawMatches(img,kp,img3,kp3,mask13,matchColor=(0,255,0), singlePointColor=(255,0,0), outImg = img3_, flags=2)
cv2.imshow('keypoints map', img3_)
cv2.waitKey(0)
cv2.imwrite('./visualization/{}.jpg'.format(counts), img3_)
counts = counts + 1


# for ii in p12:
#     print(ii.confidence)
#     print(ii.num_inliers)

# print('========================')

# for ii in p23:
#     print(ii.confidence)
#     print(ii.num_inliers)

# print('========================')

# for ii in p13:
#     print(ii.confidence)
#     print(ii.num_inliers)
#     print(ii.getInliers())
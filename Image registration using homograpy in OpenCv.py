# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:24:04 2021

@author: abc
"""
#Steps for registration using homograpy
"""
1. Import 2 images
2. Convert to grayt scale
3. Initiate ORB scale
4. Find Key points and describe them
5. Match keypoints - Brute force matcher
6. RANSAC (reject bad keypoints)
7. Register two image (use homology) 




"""
import cv2
import numpy as np

im1 = cv2.imread("monkey_distorted.jpg")
#Above image to be registered
im2 = cv2.imread("monkey.jpg")
#Above image for reference image

img1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

#Initiate ORB

orb = cv2.ORB_create(50)

kp1 , des1 = orb.detectAndCompute(img1,None)
kp2 , des2 = orb.detectAndCompute(img2,None)

# Match keypoints - Brute force matcher

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#match the descriptors

matches = matcher.match(des1,des2,None)

matches = sorted(matches, key = lambda x:x.distance)

#RANSAC (reject bad keypoints)       It's called as RANdom SAmple Consensus

points1 = np.zeros((len(matches),2), dtype=np.float32)
points2 = np.zeros((len(matches),2), dtype=np.float32)
                   
for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt                #gives index of the descriptors
    points2[i, :] = kp2[match.trainIdx].pt

h, mask = cv2.findHomography(points1,points2, cv2.RANSAC)

# Use Homography

height,width,channels = im2.shape

im1Reg = cv2.warpPerspective(im1, h, (width, height))

img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None)

cv2.imshow("Keypoint matches",img3)
cv2.imshow("Registered image",im1Reg)
cv2.waitKey(0)



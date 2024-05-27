#!/usr/bin/env python
# coding: utf-8

# Face detection and Face recognition in Open CV

# In[1]:


pip install opencv-contrib-python


# In[2]:


pip install caer


# In[ ]:


import cv2 as cv

img = cv.imread('taylor_swift.jpg')
cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray person', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness = 2)
    
cv.imshow('Detected Faces', img)    

cv.waitKey(0)


# In[ ]:


import cv2 as cv

img = cv.imread('group_photo.jpg')
cv.imshow('Persons', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Gray person', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness = 2)
    
cv.imshow('Detected Faces', img)    

cv.waitKey(0)


# In[1]:


pip install canaro


# In[ ]:





import os
from os import listdir
from PIL import Image
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras_facenet import FaceNet
import numpy as np

import tensorflow as tf

import pickle 
import cv2

HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()

folder = 'pictures'
database = {}
pathlist= os.listdir(folder)



for path in pathlist:
    # print(filename)
    img = cv2.imread(os.path.join(folder,path))
    
    img1 = HaarCascade.detectMultiScale(img,1.1,4)
    
    if len(img1)>0:
        x1, y1, width, height = img1[0]         
    else:
        x1, y1, width, height = 1, 1, 10, 10
        
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2)                 
    img_array = asarray(img2)
    
    face = img_array[y1:y2, x1:x2]                        
    
    face = Image.fromarray(face)                       
    face = face.resize((160,160))
    face = asarray(face)
    
    face = expand_dims(face, axis=0)
    signature = MyFaceNet.embeddings(face)
    
    database[os.path.splitext(path )[0]]=signature
    
myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()

print(database)
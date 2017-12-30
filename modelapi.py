from keras.models import load_model
import h5py


#model.load_weights('my_model_weights.h5', by_name=True)

from scipy.ndimage import imread
from scipy.misc import imresize
from skimage import color
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np


import os
from os import listdir
from os.path import isfile, join


model = load_model('my_final_model.h5')
print('yeapiiiiii')

def predict(imgpath):
	#i = random.randint(0,len(var1)-1)
    #filename=os.path.join(self.dir,self.filenames[i])
    #filename=os.path.join(dir1,var1[i])
	img = cv2.imread(imgpath, 0)
	img= color.rgb2gray(img)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,99,10)
	kernel = np.ones((3,3),np.uint8)
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	#img = cv2.adaptiveThreshold
	#(closing,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,99,10)
	#gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
	#img = imresize(closing,[220,220])
	shape = img.shape
	if(shape[1]<220):
		top=(220-(shape[1]))//2
	else:
		top=0
	#bottom
	if(shape[0]<220):
		left=(220-(shape[0]))//2
	else:
		left=0
    #right
    #img1=[][][]
	constant= cv2.copyMakeBorder(closing,top,top,left,left,cv2.BORDER_CONSTANT,value=255)
    #img = cv2.resize(constant,(220,220))
	img = cv2.resize(constant,(96,96))
    #blur
    #constant= cv2.resize(constant, (32, 32))
    #img= cv2.GaussianBlur(img,(3,3),0)
    #kernel = np.ones((3,3),np.uint8)
	#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	img= img/255
	img=np.expand_dims(img,axis=3)
	#x=np.array(x)
	#img =np.reshape(img,(1,96,96,1) )
	imgn=np.asarray(img)
	preds=model.predict(np.array([imgn]))[0]
	#print(preds.shape)
	#print(preds)
	answer=[]
	#for j in range(0,80):
	#	if(preds[j]>0.5):
	#		answer.append(labels[j])
	answer = np.where(preds>0.5)[0]
	#print(answer)
	return answer
	

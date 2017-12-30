import cv2
import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
from skimage import color
import random
import pickle
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout,GlobalAveragePooling2D, Flatten, Dense, Reshape, BatchNormalization, Activation, MaxPooling2D,InputLayer
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.layers import Lambda
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model

var = os.listdir("train_images") #listing all names of directory

#getting list of unicodes for each file
unicode = []
for i in var:
    temp = i.split(".")
    temp = temp[0].split("_")
    unicode.append(temp[3:])
print(unicode)

"""
it takes maximum length of it
maxt = len(unicode[0])
for i in unicode:
    if maxt<len(i) :
        maxt = len(i)
print(maxt)"""

#generating data in form of filename and unicodes associated to it
data=[]
for i,j in enumerate(var):
    data.append([j,unicode[i]])
print(data)

#pickling
#pickle.dump( data, open( 'data.pk', 'wb' ) )
save = pickle.load(open('data.pk','rb'))


dir1="./train_images/"
max_output_digits = 5
NUM_OUTPUT_DIGITS=5
def preproc(imgpath):
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
    return img    


#splitting data into train and validation
split=len(data)//10
random.shuffle(data)
listTrain=data[split:]
listVal=data[:split]

#sanity check
for f in range(1):
	imgname = listTrain[f][0]
	filepath = os.path.join(dir1,imgname)
	print(filepath)
	img = preproc(filepath)
	plt.imshow(img)
	plt.show()
	print(listTrain[f][1])
	print(img.shape)

def LecunLCN(X, image_shape=(None,96,96,1), threshold=1e-4, radius=7, use_divisor=True):
    """Local Contrast Normalization"""
    """[http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf]"""

    # Get Gaussian filter
    filter_shape = (radius, radius, image_shape[3], 1)

    filters = gaussian_filter(filter_shape)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    # Compute the Guassian weighted average by means of convolution
    convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')

    # Subtractive step
    mid = int(np.floor(filter_shape[1] / 2.))

    # Make filter dimension broadcastable and subtract
    centered_X = tf.subtract(X, convout)

    # Boolean marks whether or not to perform divisive step
    if use_divisor:
        # Note that the local variances can be computed by using the centered_X
        # tensor. If we convolve this with the mean filter, that should give us
        # the variance at each point. We simply take the square root to get our
        # denominator

        # Compute variances
        sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')

        # Take square root to get local standard deviation
        denom = tf.sqrt(sum_sqr_XX)

        per_img_mean = tf.reduce_mean(denom)
        divisor = tf.maximum(per_img_mean, denom)
        # Divisise step
        new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
    else:
        new_X = centered_X

    return new_X

def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)
    
    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)
    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)

def gauss(x, y, sigma=3.0):
    Z = 2 * np.pi * sigma ** 2
    return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))


#model	
model = Sequential()		

model.add(InputLayer(input_shape=(96,96,1)))
#model.add(Lambda(LecunLCN))

model.add(Conv2D(48, 3, strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, 3, strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
    
model.add(Conv2D(64, 3, strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(128, 1, strides=(1,1), padding='valid'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))


'''model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))'''

model.add(Dense(256, activation='relu'))
#model.add(Dense(NUM_OUTPUT_DIGITS * 11))
#model.add(Reshape((NUM_OUTPUT_DIGITS, 11)))
model.add(Dense(128))
#model.add(Reshape((NUM_OUTPUT_DIGITS, 11)))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())

#forming data set to be trained on.
dataset=[]
labels=[]


'''def flat_one_hot(label):
        label = np.sort(label,axis=None)
        x = np.zeros((1,128),dtype= int)
        for i in label:
                x[i-2303]=1
        return x'''

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
#values = array(data)
#print(values)
# integer encode
l1=[]
for i in range(2304,2431):
	l1.append(i)
'''label_encoder = LabelEncoder()
values=array(l1)
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)
'''
#label_matrix = np.zeros((len(listTrain),max_len),dtype=int)
matrix = np.zeros((len(listTrain),128),dtype=int)







'''
le = preprocessing.LabelEncoder()
le.fit(l1)
enc=OneHotEncoder()
'''







for f in range(len(listTrain)):
	imgname = listTrain[f][0]
	filepath = os.path.join(dir1,imgname)
	print(filepath)
	img = preproc(filepath)
	#plt.imshow(img)
	#plt.show()
	#print(listTrain[f][1])
	#print(img.shape)
	labels=listTrain[f][1]
	intlab=[]
	for i in range(len(labels)):
		intlab.append(int(labels[i]))
	intlab = np.sort(intlab,axis=None)
	#intlab=le.transform(intlab)
	for i in intlab:
		matrix[f][i-2304]=1
	dataset.append([img,intlab])
plt.imshow(dataset[0][0])
plt.show()
print(dataset[0][1])
print(dataset[0][0].shape)
print(matrix[0])
#pickle.dump(dataset, 'datasetm.pk', fix_imports=True)
#pickle.dump( dataset, open( 'datasetm.pk', 'wb' ) )
print("finally pickled! aha!")
#dataset = pickle.load(open('datasetm.pk','rb'))

#converting data to pass through model as np array
imgdata=[]
lab=[]


for f in range(len(dataset)):
	x = dataset[f][0]
	img=np.expand_dims(x,axis=3)
	#x=np.array(x)
	#img =np.reshape(img,(1,96,96,1) )
	imgn=np.asarray(img)
	imgdata.append(imgn)
	y=dataset[f][1]
	#y =np.reshape(y,(5,1) )
	lab.append(y)
	
imgndata=np.asarray(imgdata)


print("dataset is made")
'''


model.fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)


'''
'''model.add(Dense(max_output_digits * 129))
model.add(Reshape((max_output_digits, 129)))
model.add(Lambda(arrange))
model.add(Activation('sigmoid'))'''







#one_hot_encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_y_1 = LabelEncoder()
#y = labelencoder_y_1.fit_transform(y)

print("ELLLOOOO")
print(imgndata.shape)
print(matrix.shape)
model.fit(imgndata, matrix, epochs = 70, batch_size = 30)


model.save('./my_final_model1.h5')

#predict
#for i in range(len(dataset)):
#	p=matrix[i]
#	#p = np.reshape(p.shape[1],p.shape[0])
#	p = p.T
#	model.fit(imgndata[i],p, epochs=5, batch_size=1,  verbose=2)














'''
def flat_one_hot(self,label):
        x = np.zeros((max_output_digits,129),dtype= int)
        label = np.sort(label,axis=None)
        for place,val in enumerate(label):
                if(val==0):
                    x[place][0]=0
                else:
                    x[place][val-2303]=1
        return x






for i in range(3):
    channels = 16 * (2 ** i)
    model.add(Conv2D(channels, 5, strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    if(i%2==0):
        model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Conv2D(32, 1, strides=(1, 1), padding='same'))

for i in range(3):
    channels = 32 * (2 ** i)
    model.add(Conv2D(channels, 3, strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    if(i%2==0):
        model.add(MaxPooling2D(pool_size=(2,2)))

    
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(NUM_OUTPUT_DIGITS * 11))
model.add(Reshape((NUM_OUTPUT_DIGITS, 11)))
model.add(Activation('softmax')'''

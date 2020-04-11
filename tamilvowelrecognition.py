# Import libraries
import os
#import cv2
import numpy as np
import matplotlib.pyplot as plt
# Import Warnings 
import warnings
warnings.filterwarnings('ignore')
#import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# Import tensorflow as the backend for Keras
from keras import backend as K
K.common.image_dim_ordering()
#K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
#from keras.callbacks import TensorBoard
# Import required libraries for cnfusion matrix
from sklearn.metrics import classification_report,confusion_matrix
#import itertools
from PIL import Image
#import theano

PATH = os.getcwd()

# Define data path

data_path = 'D:/NAVEEN/TAMIL HANDWRITTEN/tamil\classified_dataset'
data_path2 = 'D:/NAVEEN/TAMIL HANDWRITTEN/tamil\processed_dataset'
data_dir_list = os.listdir(data_path)
data_dir_list

img_rows=128
img_cols=128

dir_path=os.path.join(data_path,'classified_dataset')
print(dir_path)

for file in data_dir_list:
    im=Image.open(data_path+'//'+file,'r')
    img=im.resize((128,128))
    gray=img.convert('L')
    gray.save(data_path2+'/'+file,"JPEG")

imlist = os.listdir(data_path2)
img_data = np.array(Image.open(data_path2 + '/'+ imlist[0]))
m,n = img_data.shape[0:2] # get the size of the images
imnbr = len(imlist) 
print(img_data.shape)


# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open(data_path2+ '\\' + img_data)).flatten() 
          for img_data in imlist],'f')
print(immatrix.shape)
        
labels=np.ones((imnbr,),dtype = int)
labels[0:277]=0    # A
labels[277:551]=1  #AA
labels[551:827]=2  # E
labels[827:1099]=3 # EE
labels[1099:1368]=4 # I
labels[1368:1629]=5 #O
labels[1629:1915]=6 #OA
labels[1915:2181]=7 #OO
labels[2181:2182]=8 #OW
labels[2182:2457]=9 #U
labels[2457:2735]=10#YE
labels[2735:2998]=11 #YEA

names = ['A','AA', 'E', 'EE', 'I', 'O', 'OA','OO','OW','U','YAE','YE']

data,Label = shuffle(immatrix,labels, random_state=2)
train_data = [data,Label]

img=immatrix[167].reshape(img_rows,img_cols)
plt.imshow(img)
#plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#%%

#batch_size to train
batch_size = 128
# number of output classes
nb_classes = 12
# number of epochs to train
nb_epoch = 2


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 1021
plt.imshow(X_train.reshape(-1,128,128)[i],interpolation='nearest')
print("label : ", Y_train[i,:])
print(X_train.shape)

#%%

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
#%%
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test)) 
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_split=0.2)


# visualizing losses and accuracy

train_loss=history.history['loss']
val_loss=history.history['val_loss']
train_acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
xc=range(nb_epoch)
#Visualizing Training Loss & Validation Loss
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#Visualizing Training Accuracy & Validation Accuracy
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%      

score = model.evaluate(X_test, Y_test,verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])
#%%
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[10].reshape(1,128,128,1))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1


#Displaying original Imag
plt.imshow(X_train[10][:,:,0]);

#Displaying above image after layer 2 .
#layer 1 is input layer 
display_activation(activations, 8, 8, 1)
#Displaying output of layer 4
display_activation(activations, 8, 8, 3)

#Displaying output of layer 8
#display_activation(activations, 8, 8, 7)

 #%%  
# Confusion Matrix

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
# (or)

y_pred = model.predict_classes(X_test)
print(y_pred)

p=model.predict_proba(X_test) # to predict probability

target_names = ['class 0(A)', 'class 1(AA)', 'class 2(E)', 'class 3(EE)', 'class 4(I)', 'class 5(O)', 'class 6(OA)', 'class 7(OW)', 'class 8(U)', 'class 9(YAE)', 'class 10(YE)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)



# Loading weights

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)


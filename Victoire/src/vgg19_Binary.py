from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint

import cv2, numpy as np
import random as rd
import os
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import matplotlib.pyplot as plt
import datetime
import time

###################################################################
# Paper source :
# Very Deep Convolutional Networks for Large-Scale Image Recognition
# K. Simonyan, A. Zisserman
# arXiv:1409.1556
# Keras code source :
# https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
###################################################################

class SupervisedDeepLearning:
    def __init__(self,weights_path=None):
        self.model = self.VGG_19(weights_path)
        rmsprop = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=1e-6)
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    def VGG_19(self,weights_path):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        if weights_path != None :
            model.load_weights(weights_path)

        return model

    def fitModel(self,batch_size=16,nb_epoch=1):
        normal_class = "cercle"
        
        X_train, Y_train, X_valid, Y_valid = self.load_data(normal_class)
        checkpointer = ModelCheckpoint(filepath='vgg19_weights.h5', verbose=1, save_best_only=True)
        history = self.model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              callbacks=[checkpointer])
        self.model.save_weights("vgg19_weights.h5")
        self.model.save("vgg19_fullmodel.h5")
        # Plot training & validation accuracy values
        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        
        plt.legend(['Train', 'Test'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Save the plot on the computer
        date = datetime.datetime.now()
        date =str(date.day)+"-"+str(date.month)+"-"+str(date.hour)+"-"+str(date.minute)    
        plt.savefig('vgg_19_fit_result_'+date+'.png')

    def load_data(self, normal_class="cat"):
        X_train = []
        Y_train = []
        X_valid = []
        Y_valid = []
        nbCercle = len(os.listdir("train/cercle"))
        nbPCercle = len(os.listdir("train/pas_cercle"))
        indC = 0
        indPC=0
        while indPC<nbCercle and indC<nbPCercle:
            r = rd.randint(0,1)
            if r==0 and indC<nbCercle:
                im = cv2.resize(cv2.imread("train/cercle/cercle_"+str(indC)+".png"), (224, 224)).astype(np.float32)
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                im = im.transpose((2,0,1))
                #im = np.expand_dims(im, axis=0)
                #im=cv2.imread("train/cercle/cercle_"+str(indC)+".png")
                r = rd.randint(0,5)
                if r == 0 :
                    X_valid.append(im)
                    Y_valid.append([1,0])
                else :
                    X_train.append(im)
                    Y_train.append([1,0])
                indC+=1
            elif indPC<nbPCercle :
                im = cv2.resize(cv2.imread("train/pas_cercle/pas_cercle_"+str(indPC)+".png"), (224, 224)).astype(np.float32)
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                im = im.transpose((2,0,1))
                #im = np.expand_dims(im, axis=0)
                #im=cv2.imread("train/pas_cercle/pas_cercle_"+str(indPC)+".png")
                r = rd.randint(0,5)
                if r == 0 :
                    X_valid.append(im)
                    Y_valid.append([0,1])
                else :
                    X_train.append(im)
                    Y_train.append([0,1])
                indPC+=1

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        X_valid = np.array(X_valid)
        Y_valid = np.array(Y_valid)

        return X_train, Y_train, X_valid, Y_valid

    def predict(self,frameSkipped):
        posX,posY = -1,-1
        debut = time.time()
        nbFrame = 0
        cap = cv2.VideoCapture("Video/test_script.mp4")
        ret, frame = cap.read()
        largeur_split = int(frame.shape[1]/3)
        hauteur_split =int(frame.shape[0]/3)

        cv2.imshow('frame',frame)
        while(True):
           
            # Capture frame-by-frame
            ret, frame = cap.read()
            mat_im = [[frame[0:hauteur_split,0:largeur_split],                frame[hauteur_split:hauteur_split*2,0:largeur_split],                 frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
                  [frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],   frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
                  [frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
            ]
            if nbFrame%frameSkipped ==0 :
                for i in range(3) :
                    for j in range(3) :      
            	    #im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
                        im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
                        im[:,:,0] -= 103.939
                        im[:,:,1] -= 116.779
                        im[:,:,2] -= 123.68
                        im = im.transpose((2,0,1))
                        im = np.expand_dims(im, axis=0)

                        out = self.model.predict(im)
                        if out[0][1] > 0.6: 
                            print("Outlier")
                    
            # Display the resulting frame
            if ret :
                nbFrame+=1
                seconds = time.time() - debut
                fps = int(nbFrame/seconds)
                cv2.putText(frame, str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
                cv2.imshow('frame',frame)

            else :
                break
            k = cv2.waitKey(1)
            if k == 27:
                break

def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)



if __name__ == "__main__":

    fit = False
    
    
    if fit :
        model = SupervisedDeepLearning()        
        model.fitModel()
    else :
        model = SupervisedDeepLearning('vgg19_weights.h5')
        model.predict(3)
        
        

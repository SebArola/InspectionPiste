from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

from sklearn.metrics import confusion_matrix
import keras
import cv2
import numpy as np
import random as rd
import os
import datetime
import time
from keras.utils import np_utils

###################################################################
# Paper source :
# Very Deep Convolutional Networks for Large-Scale Image Recognition
# K. Simonyan, A. Zisserman
# arXiv:1409.1556
# Keras code source :
# https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
###################################################################

class VGG_Binary:

	##
	# __init__ :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : initialise the model with sgd optimizer
	## 
	def __init__(self,vgg,weights_path=None ):
		if vgg == 19:
			self.model = self.VGG_19(weights_path)
		else:
			self.model = self.VGG_16(weights_path)
		rmsprop = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=1e-6)
		sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

	##
	# VGG_19 :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : create the vgg neural network. Source in the header
	## 
	def VGG_19(self,weights_path):

		if weights_path != None :
			model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling=None, classes=1000)
			for layer in model.layers:
				layer.trainable = False

		x = model.output
		x = Flatten()(x)

		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)
		x = Dense(4096, activation='relu')(x) 
		x = Dropout(0.5)(x)

	    # New softmax layer
		predictions = Dense(1, activation='sigmoid')(x) 

		finetune_model = Model(inputs=model.input, outputs=predictions)

		return finetune_model

	
	def VGG_16(self, weights_path):
		my_VGG16 = Sequential() #Creation d'un reseau de neurones vide


		my_VGG16.add(Conv2D(64, (3, 3), input_shape=(3,224, 224), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format="channels_first"))

		my_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format="channels_first"))

		my_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format="channels_first"))

		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format="channels_first"))

		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
		my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), data_format="channels_first"))

		my_VGG16.add(Flatten()) #Conversion des matrices 3D en vecteur 1D

		#Ajout de la premiere couche fully connected, suivie d'une couche ReLU
		my_VGG16.add(Dense(4096, activation='relu'))

		#Ajout de la deuxieme couche fully connected, suivie d'une couche ReLU
		my_VGG16.add(Dense(4096, activation='relu'))

		#Ajout de la derniere couche fully connected qui permet de classifier
		my_VGG16.add(Dense(1, activation='sigmoid'))

		if weights_path != None :
			my_VGG16.load_weights(weights_path)

		return my_VGG16
	##
	# fitModel :
	# 	input :
	#		batch_size : size of the batch (default 16)
	#		nb_epoch   : number of epoch (default 1)
	#	Descrtiption : fit the vgg neural network with the given parameters and the labelled data base
	## 
	def fitModel(self, weigt_name,batch_size=16,nb_epoch=1):
		
		X_train, Y_train, X_valid, Y_valid = self.load_data()
		checkpointer = ModelCheckpoint(filepath=weigt_name, verbose=1, save_best_only=True)
		earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
		history = self.model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=nb_epoch,
			  shuffle=True,
			  verbose=1,
			  validation_data=(X_valid, Y_valid),
			  callbacks=[checkpointer])
		self.model.save_weights(weigt_name)
		#self.model.save("vgg19_fullmodel.h5")
		prediction = self.model.predict(X_valid,batch_size=batch_size, verbose=1)
		print(prediction)
		#y_classes = keras.utils.np_utils.probas_to_classes(prediction)
		cm = confusion_matrix(Y_valid, prediction)
		return (history,cm)
	   
	##
	# load_data :
	#	input :
	#		normalPath : path to the runway pictures with no debris (default "train/normal")
	#		debrisPath   : path to the runway pictures with no debris (default "train/debris")
	#	Descrtiption : Load the data in the data base, format need to respected 
	#	DataBase format : piste_1.png debris_1.png
	##
	def load_data(self, normalPath="../Data/train/normal", debrisPath="../Data/train/debris", validNormalPath = "../Data/valid/normal", validDebrisPath = "../Data/valid/debris"):
		X_trainTab = []
		Y_trainTab = []
		X_validTab = []
		Y_validTab = []
		nbNormal = len(os.listdir(normalPath))
		nbDebris = len(os.listdir(debrisPath))
		nbValidNormal = len(os.listdir(validNormalPath))
		nbValidDebris = len(os.listdir(validDebrisPath))
		indN = 0
		indD= 0
		indVN = 3254 #Suite a une erreur durant l'annotation 3254
		indVD = 0
		while indD<nbDebris or indN<nbNormal:
			r = rd.randint(0,1)
			if r==0 and indN<nbNormal:
				while os.path.isfile(normalPath+"/piste_"+str(indN)+".png") == False:
					indN+=1
					nbNormal+=1
				img_path = normalPath+"/piste_"+str(indN)+".png"
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				#x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)

				X_trainTab.append(x)
				Y_trainTab.append(0)
				indN+=1
			if r==1 and indD<nbDebris :
				while os.path.isfile(debrisPath+"/debris_"+str(indD)+".png") == False:
					indD+=1
					nbDebris+=1
				img_path = debrisPath+"/debris_"+str(indD)+".png"
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				#x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)

				X_trainTab.append(x)
				Y_trainTab.append(1)
				indD+=1

		while indVD<nbValidDebris or indVN<nbValidNormal+3254:
			r = rd.randint(0,1)
			if r==0 and indVN < nbValidNormal+3254:
				while os.path.isfile(validNormalPath+"/piste_"+str(indVN)+".png") == False:
					indVN+=1
					nbValidNormal+=1
				img_path = validNormalPath+"/piste_"+str(indVN)+".png"
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				#x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)

				X_validTab.append(x)
				Y_validTab.append(0)

				indVN+=1
			if r==1 and indVD < nbValidDebris :
				while os.path.isfile(validDebrisPath+"/debris_"+str(indVD)+".png") == False:
					indVD+=1
					nbValidDebris+=1
				img_path = validDebrisPath+"/debris_"+str(indVD)+".png"
				img = image.load_img(img_path, target_size=(224, 224))
				x = image.img_to_array(img)
				#x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)

				X_validTab.append(x)
				Y_validTab.append(1)
				
				indVD+=1
		print(len(X_trainTab))
		print("indD  :"+str(indD))
		print("indN  :"+str(indN))
		print("indVD :"+str(indVD))
		print("indVN :"+str(indVN))
		X_train = np.array(X_trainTab)
		Y_train = np.array(Y_trainTab)

		X_valid = np.array(X_validTab)
		Y_valid = np.array(Y_validTab)

		return X_train, Y_train, X_valid, Y_valid

	def load_data_for_predict(self, normalPath="../Data/train/normal", debrisPath="../Data/train/debris"):
		X_trainTab = []
		Y_trainTab = []
		
		nbNormal = len(os.listdir(normalPath))
		nbDebris = len(os.listdir(debrisPath))
		indN = 0
		indD= 0
		while indD<nbNormal and indN<nbDebris:
			r = rd.randint(0,1)
			if r==0 and indN<nbNormal:
				while os.path.isfile(normalPath+"/piste_"+str(indN)+".png") == False:
					indN+=1
				im = cv2.imread(normalPath+"/piste_"+str(indN)+".png")
				
				X_trainTab.append(im)
				Y_trainTab.append(1)
				
				indN+=1
			elif indD<nbDebris :
				while os.path.isfile(debrisPath+"/debris_"+str(indD)+".png") == False:
					indD+=1

				im = cv2.imread(debrisPath+"/debris_"+str(indD)+".png")
				
				X_trainTab.append(im)
				Y_trainTab.append(0)
				
				indD+=1		

		return X_trainTab, Y_trainTab
	##
	# predict :
	#	input :
	#		frameSkipped : number of frame to skipped
	#		videoInput : path to the video input
	#	Descrtiption : make a prediction on the frame from videoInput
	##	
	def lol_predict(self,frameSkipped, videoInput):
		X_train, Y_train = self.load_data_for_predict()
		
		for im in X_train :
			
			frame = im
			debris =False
			im = cv2.resize(np.array(im), (224, 224)).astype(np.float32)
			im[:,:,0] -= 103.939
			im[:,:,1] -= 116.779
			im[:,:,2] -= 123.68
			im = im.transpose((2,0,1))
			im = np.expand_dims(im, axis=0)
			out = self.model.predict(im)
			if out[0][0] > 0.8:
				debris=True
			cv2.putText(frame, "Debris : "+ str(debris), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
			print(debris)
			cv2.imshow('image',frame)
			k = cv2.waitKey(1)
			while k!=32:
				k = cv2.waitKey(1)
		
	   

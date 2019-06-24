from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
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
		sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

	##
	# VGG_19 :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : create the vgg neural network. Source in the header
	## 
	def VGG_19(self,weights_path):

		# model = Sequential()
		# model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
		# model.add(Conv2D(64, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(64, (3, 3), activation="relu"))
		# model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(128, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(128, (3, 3), activation="relu"))
		# model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(256, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(256, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(256, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(256, (3, 3), activation="relu"))
		# model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(ZeroPadding2D((1,1)))
		# model.add(Conv2D(512, (3, 3), activation="relu"))
		# model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first"))

		

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
		history = self.model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=nb_epoch,
			  shuffle=True,
			  verbose=1,
			  validation_data=(X_valid, Y_valid),
			  callbacks=[checkpointer])
		self.model.save_weights(weigt_name)
		#self.model.save("vgg19_fullmodel.h5")
		prediction = self.model.predict_classes(X_valid,batch_size=batch_size, verbose=1)
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
	def load_data(self, normalPath="../Data/train/normal", debrisPath="../Data/train/debris"):
		X_trainTab = []
		Y_trainTab = []
		X_validTab = []
		Y_validTab = []
		nbNormal = len(os.listdir(normalPath))
		nbDebris = len(os.listdir(debrisPath))
		indN = 0
		indD= 0
		while indD<nbNormal and indN<nbDebris:
			r = rd.randint(0,1)
			if r==0 and indN<nbNormal:
				while os.path.isfile(normalPath+"/piste_"+str(indN)+".png") == False:
					indN+=1
				im = cv2.resize(cv2.imread(normalPath+"/piste_"+str(indN)+".png"), (224, 224)).astype(np.float32)
				# im[:,:,0] -= 103.939
				# im[:,:,1] -= 116.779
				# im[:,:,2] -= 123.68
				#im = im.transpose((2,0,1))
				#im = np.expand_dims(im, axis=0)
				#im=cv2.imread("train/cercle/cercle_"+str(indN)+".png")
				r = rd.randint(0,8)
				if r == 0 :
					X_validTab.append(im)
					Y_validTab.append(1)
				else :
					X_trainTab.append(im)
					Y_trainTab.append(1)
				indN+=1
			elif indD<nbDebris :
				while os.path.isfile(debrisPath+"/debris_"+str(indD)+".png") == False:
					indD+=1

				im = cv2.resize(cv2.imread(debrisPath+"/debris_"+str(indD)+".png"), (224, 224)).astype(np.float32)
				# im[:,:,0] -= 103.939
				# im[:,:,1] -= 116.779
				# im[:,:,2] -= 123.68
				#im = im.transpose((2,0,1))
				#im = np.expand_dims(im, axis=0)
				#im=cv2.imread("train/pas_cercle/pas_cercle_"+str(indD)+".png")
				r = rd.randint(0,5)
				if r == 0 :
					X_validTab.append(im)
					Y_validTab.append(0)
				else :
					X_trainTab.append(im)
					Y_trainTab.append(0)
				indD+=1

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
		
	   

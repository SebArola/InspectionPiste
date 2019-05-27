from keras.models import Sequential, Model
from keras.layers import Input, Activation, Concatenate, GlobalAveragePooling2D

from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np
import random as rd
import os
import datetime
import time
from keras.utils import np_utils

###################################################################
# Paper source :
# Keras code source :
# https://github.com/DT42/squeezenet_demo
###################################################################

class SqueezeNet:

	##
	# __init__ :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : initialise the model with sgd optimizer
	## 
	def __init__(self,weights_path=None):
		self.model = self.SqueezeNet()
		rmsprop = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=1e-6)
		sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

	##
	# VGG_19 :
	# 	input :
	#		weights_path : the path to the trained weight, none if there is no weight
	#	Descrtiption : create the vgg neural network. Source in the header
	## 
	def SqueezeNet(nb_classes=1, inputs=(3, 224, 224)):
		""" Keras Implementation of SqueezeNet(arXiv 1602.07360)
		@param nb_classes: total number of final categories
		Arguments:
		inputs -- shape of the input images (channel, cols, rows)
		"""
		input_img = Input(shape=(3, 224, 224))
		conv1 = Conv2D(
			96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
			strides=(2, 2), padding='same', name='conv1',
			data_format="channels_first")(input_img)
		maxpool1 = MaxPooling2D(
			pool_size=(3, 3), strides=(2, 2), name='maxpool1',
			data_format="channels_first")(conv1)
		fire2_squeeze = Conv2D(
			16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire2_squeeze',
			data_format="channels_first")(maxpool1)
		fire2_expand1 = Conv2D(
			64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire2_expand1',
			data_format="channels_first")(fire2_squeeze)
		fire2_expand2 = Conv2D(
			64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire2_expand2',
			data_format="channels_first")(fire2_squeeze)
		merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

		fire3_squeeze = Conv2D(
			16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire3_squeeze',
			data_format="channels_first")(merge2)
		fire3_expand1 = Conv2D(
			64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire3_expand1',
			data_format="channels_first")(fire3_squeeze)
		fire3_expand2 = Conv2D(
			64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire3_expand2',
			data_format="channels_first")(fire3_squeeze)
		merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

		fire4_squeeze = Conv2D(
			32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire4_squeeze',
			data_format="channels_first")(merge3)
		fire4_expand1 = Conv2D(
			128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire4_expand1',
			data_format="channels_first")(fire4_squeeze)
		fire4_expand2 = Conv2D(
			128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire4_expand2',
			data_format="channels_first")(fire4_squeeze)
		merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
		maxpool4 = MaxPooling2D(
			pool_size=(3, 3), strides=(2, 2), name='maxpool4',
			data_format="channels_first")(merge4)

		fire5_squeeze = Conv2D(
			32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire5_squeeze',
			data_format="channels_first")(maxpool4)
		fire5_expand1 = Conv2D(
			128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire5_expand1',
			data_format="channels_first")(fire5_squeeze)
		fire5_expand2 = Conv2D(
			128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire5_expand2',
			data_format="channels_first")(fire5_squeeze)
		merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

		fire6_squeeze = Conv2D(
			48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire6_squeeze',
			data_format="channels_first")(merge5)
		fire6_expand1 = Conv2D(
			192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire6_expand1',
			data_format="channels_first")(fire6_squeeze)
		fire6_expand2 = Conv2D(
			192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire6_expand2',
			data_format="channels_first")(fire6_squeeze)
		merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

		fire7_squeeze = Conv2D(
			48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire7_squeeze',
			data_format="channels_first")(merge6)
		fire7_expand1 = Conv2D(
			192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire7_expand1',
			data_format="channels_first")(fire7_squeeze)
		fire7_expand2 = Conv2D(
			192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire7_expand2',
			data_format="channels_first")(fire7_squeeze)
		merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

		fire8_squeeze = Conv2D(
			64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire8_squeeze',
			data_format="channels_first")(merge7)
		fire8_expand1 = Conv2D(
			256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire8_expand1',
			data_format="channels_first")(fire8_squeeze)
		fire8_expand2 = Conv2D(
			256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire8_expand2',
			data_format="channels_first")(fire8_squeeze)
		merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

		maxpool8 = MaxPooling2D(
			pool_size=(3, 3), strides=(2, 2), name='maxpool8',
			data_format="channels_first")(merge8)
		fire9_squeeze = Conv2D(
			64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire9_squeeze',
			data_format="channels_first")(maxpool8)
		fire9_expand1 = Conv2D(
			256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire9_expand1',
			data_format="channels_first")(fire9_squeeze)
		fire9_expand2 = Conv2D(
			256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
			padding='same', name='fire9_expand2',
			data_format="channels_first")(fire9_squeeze)
		merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

		fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
		conv10 = Conv2D(
			1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
			padding='valid', name='conv10',
			data_format="channels_first")(fire9_dropout)

		global_avgpool10 = GlobalAveragePooling2D(data_format='channels_first')(conv10)
		sigmoid = Activation("sigmoid", name='sigmoid')(global_avgpool10)

		return Model(inputs=input_img, outputs=sigmoid)

	##
	# fitModel :
	# 	input :
	#		batch_size : size of the batch (default 16)
	#		nb_epoch   : number of epoch (default 1)
	#	Descrtiption : fit the vgg neural network with the given parameters and the labelled data base
	## 
	def fitModel(self,batch_size=16,nb_epoch=1):
		
		X_train, Y_train, X_valid, Y_valid = self.load_data()
		checkpointer = ModelCheckpoint(filepath='squeezenet_weights_sig.h5', verbose=1, save_best_only=True)
		history = self.model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=nb_epoch,
			  shuffle=True,
			  verbose=1,
			  validation_data=(X_valid, Y_valid),
			  callbacks=[checkpointer])
		self.model.save_weights("squeezenet_weights_sig.h5")
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
				im[:,:,0] -= 103.939
				im[:,:,1] -= 116.779
				im[:,:,2] -= 123.68
				im = im.transpose((2,0,1))
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
				im[:,:,0] -= 103.939
				im[:,:,1] -= 116.779
				im[:,:,2] -= 123.68
				im = im.transpose((2,0,1))
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
	def predict(self,frameSkipped, videoInput):
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
		# debut = time.time()
		# nbFrame = 0
		# cap = cv2.VideoCapture(videoInput)
		# ret, frame = cap.read()
		# largeur_split = int(frame.shape[1]/3)
		# hauteur_split =int(frame.shape[0]/3)

		# cv2.imshow('frame',frame)
		# while(True):
		  
		# 	# Capture frame-by-frame
		# 	ret, frame = cap.read()
		# 	mat_im = [[frame[0:hauteur_split,0:largeur_split],				frame[hauteur_split:hauteur_split*2,0:largeur_split],				 frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
		# 		  [frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],   frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
		# 		  [frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
		# 	]
		# 	nbDebris = 0
		# 	if nbFrame%frameSkipped ==0 :
		# 		for i in range(3) :
		# 			for j in range(3) :	  
		# 			#im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
		# 				im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
		# 				im[:,:,0] -= 103.939
		# 				im[:,:,1] -= 116.779
		# 				im[:,:,2] -= 123.68
		# 				im = im.transpose((2,0,1))
		# 				im = np.expand_dims(im, axis=0)

		# 				out = self.model.predict(im)
		# 				#print(out)
		# 				if out[0][0] > 0.8:
		# 					nbDebris+=1
		# 					print("Outlier")
					
		# 	# Display the resulting frame
		# 	if ret :
		# 		nbFrame+=1
		# 		seconds = time.time() - debut
		# 		fps = int(nbFrame/seconds)
		# 		cv2.putText(frame, str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
		# 		cv2.putText(frame, "Nb Debris : "+ str(nbDebris), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
		# 		cv2.imshow('frame',frame)

		# 	else :
		# 		break
		# 	k = cv2.waitKey(1)
		# 	if k == 27:
		# 		break
	   

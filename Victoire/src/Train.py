from vgg_Binary import VGG_Binary
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix
import numpy as np
from SqueezeNet import SqueezeNet
import datetime
import time
import cv2

###################################
# Main.py : 
# 	Author : Sébastien Arola
#	Description : main class
###################################

##
# plotHistory :
# 	input :
#		history : the history to be plot
#	Descrtiption : plot the given history
## 
def plotHistory(history, ficName):
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

		#Save the plot on the computer
		date = datetime.datetime.now()
		date =str(date.day)+"-"+str(date.month)+"-"+str(date.hour)+"-"+str(date.minute)	
		plt.savefig(ficName+date+'.png')

		#Display the plot
		plt.show()



def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()
	return ax

##
# predict :
#	input :
#		frameSkipped : number of frame to skipped
#		videoInput : path to the video input
#	Descrtiption : make a prediction on the frame from videoInput
##	
def predict(frameSkipped, videoInput, model):
	debut = time.time()
	nbFrame = 0
	cap = cv2.VideoCapture(videoInput)
	ret, frame = cap.read()
	largeur_split = int(frame.shape[1]/3)
	hauteur_split =int(frame.shape[0]/3)

	cv2.imshow('frame',frame)
	while(True): 
		#Capture frame-by-frame
		ret, frame = cap.read()
		mat_im = [[frame[0:hauteur_split,0:largeur_split],				frame[hauteur_split:hauteur_split*2,0:largeur_split],				 frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
			 		  [frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],   frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
			 		  [frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
		 	]
		
		if nbFrame%frameSkipped ==0 :
			debris = False
			for i in range(3) :
				for j in range(3) :	  
					im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
					# im[:,:,0] -= 103.939
					# im[:,:,1] -= 116.779
					# im[:,:,2] -= 123.68
					im = im.transpose((2,0,1))
					im = np.expand_dims(im, axis=0)

					out = model.predict(im)
					print(out)
					if out[0][0] == 0:
						debris= True
						#print("Outlier")

					
		# Display the resulting frame
		if ret :
			nbFrame+=1
			seconds = time.time() - debut
			fps = int(nbFrame/seconds)
			cv2.putText(frame, str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
			cv2.putText(frame, "Nb Debris : "+ str(debris), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
			cv2.imshow('fromame',frame)

		else :
			break
		k = cv2.waitKey(1)
		if k == 27:
			break

if __name__ == "__main__":

	fit = True
	
	
	if fit :
		model = VGG_Binary(19,"vgg19_weights.h5")
		#model = SqueezeNet()		
		history = model.fitModel("vgg19_transfer.h5",12,20)
		plotHistory(history[0],'vgg19_fit_result_')
		cm_plot_labels = ["no_debris","debris"]
		plot_confusion_matrix(history[1], cm_plot_labels,normalize=True,title="Confusion Matrix")
	else :
		model = VGG_Binary(19,'vgg19_weights_sig.h5')
		#model = VGG_Binary(16,'vgg16_sig_20_epoch.h5')
		predict(3,"../../Script/debris_parking_4.mp4", model.model)
		
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
#   Inspection.py : 
# 	Author : Sébastien Arola
#	Description : main class
###################################

def debris_or_not_debris(sortie,seuil):
	if sortie[0][0] > seuil:
		return True
	else :
		return False


##
# predict :
#	input :
#		frameSkipped : number of frame to skipped
#		videoInput : path to the video input
#	Descrtiption : make a prediction on the frame from videoInput
##	
def predict(frameSkipped, videoInput, model, seuil):
	debut = time.time()
	nbFrame = 0
	cap = cv2.VideoCapture(videoInput)
	ret, frame = cap.read()
	largeur_split = int(frame.shape[1]/3)
	hauteur_split =int(frame.shape[0]/3)

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
					# im[:,:,2] -= 123.68n
					#im = im.transpose((2,0,1))
					im = np.expand_dims(im, axis=0)

					out = model.predict(im)
					print(out)
					if debris_or_not_debris(out, seuil):
						debris= True

						break;
		if debris:
		#arrêt du drone et verif			
			print(debris)
		else :
			print(debris)
		k = cv2.waitKey(1)
		if k == 27:
			break

if __name__ == "__main__":

	weigt = "vgg19_transfer.h5"
	cam = "../../Script/debris_parking_4.mp4"
	#model = VGG_19_Binary('vgg19_weights_sig.h5')
	model = VGG_Binary(19,weigt)
	predict(3,cam, model.model, 0.8)
		
from vgg_Binary import VGG_Binary
import cv2
import numpy as np

class Video :

	def __init__(self, weight, seuil, video_url):
		self.model = VGG_Binary(weight)
		self.seuil = seuil
		self.video = video_url

	def debris_or_not_debris(self,sortie):
		if sortie[0][0] > self.seuil:
			return True
		else :
			return False

	##
	# predict :
	#	Descrtiption : make a prediction on the frame from self.video, called by Main.py
	##	
	def predict(self):

		cap = cv2.VideoCapture(self.video)
		ret, frame = cap.read()
		largeur_split = int(frame.shape[1]/3)
		hauteur_split =int(frame.shape[0]/3)

		mat_im = [[frame[0:hauteur_split,0:largeur_split],				  frame[hauteur_split:hauteur_split*2,0:largeur_split],				   frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
				 [ frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],  frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
				 [ frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
			 	]
			
		debris = False
		for i in range(3) :
			for j in range(3) :	  
				im = cv2.resize( np.array(mat_im[i][j]), (224, 224)).astype(np.float32)
				im = np.expand_dims(im, axis=0)

				out = self.model.predict(im)
				if debris_or_not_debris(out):
					return True
		return False
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix
import datetime
import time
import cv2
from Video import Video

###################################
#   Inspection.py : 
# 	Author : Sébastien Arola
#	Description : main class
###################################

class Main:

	def __init__(self):
		(weight, seuil, video_url, tAttente) = self.getConfigInfo()
		self.temps_attente = tAttente
		self.video = Video(weight, seuil, video_url)

	def getConfigInfo(self):
		weigt = "vgg19_transfer.h5"
		cam = "../../Script/debris_parking_3.mp4"
		seuil = 0.8
		tAttente = 2
		return (weigt,seuil,cam, tAttente)

	def startPredict(self):
		check = time.time()
		while(True):
			if time.time()-check >= self.temps_attente:
				tPreTraitement =time.time()
				print("\n++++++++\nTemps d'attente : "+str(int(time.time()-check)))
				result = self.video.predict()
				print("Duree traitement : " + str(time.time()-tPreTraitement))
				print("Debris : " + str(result))
				check = time.time()

if __name__ == "__main__":
	main = Main()
	main.startPredict()
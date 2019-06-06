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
		config = []
		nb_param = 4
		header = True
		fichier = open("config.txt", "r")
		for ligne in fichier:
			if header == False :
				if ligne != "}":
					config.append(ligne.split("=")[1].split(";")[0])
			if ligne == "{\n" :
				header = False
			
		if len(config)<nb_param :
			print("Error : expected "+str(nb_param)+" parameters in config.txt, but found "+str(len(config))+".")
			exit(0)
		else :
			weight    = config[0]
			cam      = config[1]
			seuil    = config[2]
			tAttente = config[3]
			return (weight,seuil,cam, tAttente)

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
	#main.startPredict()
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix
from SqueezeNet import SqueezeNet
import datetime
import time
import cv2

###################################
#   Inspection.py : 
# 	Author : Sébastien Arola
#	Description : main class
###################################






if __name__ == "__main__":

	weigt = "vgg19_transfer.h5"
	cam = "../../Script/debris_parking_4.mp4"
	#model = VGG_19_Binary('vgg19_weights_sig.h5')
	model = VGG_Binary(19,weigt)
	predict(3,cam, model.model, 0.8)
		
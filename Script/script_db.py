import cv2
import tkinter as tk
import numpy as np

mouseY, mouseX = 0,0

def click_debris(event,x,y,flags,param):
	global mouseY, mouseX
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX,mouseY = x,y

def wich_case(x,y,xmax,ymax):
	posX = int(3 * x/xmax)
	posY = int(3 * y/ymax)
	return (posX, posY)

	
numB = 0
numPB = 0 
posX,posY = -1,-1
cap = cv2.VideoCapture("../Video/test_script.mp4")
ret, frame = cap.read()
largeur_split = int(frame.shape[1]/3)
hauteur_split =int(frame.shape[0]/3)

cv2.imshow('frame',frame)
cv2.setMouseCallback('frame',click_debris)
while(True):
	k = cv2.waitKey(1)
	if k==32 :
		mat_im = [[frame[0:hauteur_split,0:largeur_split],                frame[hauteur_split:hauteur_split*2,0:largeur_split],                 frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
				  [frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],   frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
				  [frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
		]
		posX,posY = -1,-1
		if mouseX !=0 :
			posX,posY = wich_case(mouseX,mouseY,frame.shape[1],frame.shape[0])	
			cv2.imwrite('../DataBase/train/Debris/im_'+str(numPB)+'.png',mat_im[posX][posY])
			mouseX = 0
			numPB+=1
		
		for i in range(3) :
			for j in range(3) :
				if i!= posX or j!= posY:			
					cv2.imwrite('../DataBase/train/Ndebris/im_'+str(numB)+'.png',mat_im[i][j])
					numB+=1

		# Capture frame-by-frame
		ret, frame = cap.read()

		
		# Display the resulting frame
		if ret :
			cv2.imshow('frame',frame)
		else :
			break

	if k == 27:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
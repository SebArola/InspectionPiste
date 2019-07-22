import cv2
import tkinter as tk
import numpy as np

mouseXY = []
nbMClick = 0

def click_debris(event,x,y,flags,param):
	global mouseXY
	global nbMClick
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseXY.append([x,y])
		nbMClick+=1

def wich_case(x,y,xmax,ymax):
	posX = int(3 * x/xmax)
	posY = int(3 * y/ymax)
	return (posX, posY)

	
numB = 3254
numPB = 3732
posX,posY = -1,-1
allPosXY = []
cap = cv2.VideoCapture("pas_debris_02.MOV")
ret, frame = cap.read()
frame = cv2.resize(frame,(1366,768))
split = 3
largeur_split = int(frame.shape[1]/split)
hauteur_split =int(frame.shape[0]/split)

cv2.line(frame,(0,hauteur_split),(frame.shape[1],hauteur_split),(0,0,255),1)
cv2.line(frame,(0,hauteur_split*2),(frame.shape[1],hauteur_split*2),(0,0,255),1)

cv2.line(frame,(largeur_split,0),(largeur_split,frame.shape[0]),(0,0,255),1)
cv2.line(frame,(largeur_split*2,0),(largeur_split*2,frame.shape[0]),(0,0,255),1)
cv2.imshow('frame',frame)
cv2.setMouseCallback('frame',click_debris)
while(True):
	k = cv2.waitKey(1)
	if k==97	:

		ret, frame = cap.read()
		frame = cv2.resize(frame,(1366,768))
		
		# Display the resulting frame
		if ret :
			cv2.line(frame,(0,hauteur_split),(frame.shape[1],hauteur_split),(0,0,255),1)
			cv2.line(frame,(0,hauteur_split*2),(frame.shape[1],hauteur_split*2),(0,0,255),1)

			cv2.line(frame,(largeur_split,0),(largeur_split,frame.shape[0]),(0,0,255),1)
			cv2.line(frame,(largeur_split*2,0),(largeur_split*2,frame.shape[0]),(0,0,255),1)
			cv2.imshow('frame',frame)

	if k==32 :
		mat_im = [[frame[0:hauteur_split,0:largeur_split],				  frame[hauteur_split:hauteur_split*2,0:largeur_split],				 frame[hauteur_split*2:frame.shape[0],0:largeur_split]],
				  [frame[0:hauteur_split,largeur_split:largeur_split*2],  frame[hauteur_split:hauteur_split*2,largeur_split:largeur_split*2],   frame[hauteur_split*2:frame.shape[0],largeur_split:largeur_split*2]],
				  [frame[0:hauteur_split,largeur_split*2:frame.shape[1]], frame[hauteur_split:hauteur_split*2,largeur_split*2:frame.shape[1]], frame[hauteur_split*2:frame.shape[0],largeur_split*2:frame.shape[1]]]
		]

		# mat_im = [[frame[0:hauteur_split,0:largeur_split],				  	frame[hauteur_split:frame.shape[0],0:largeur_split]],
		# 		  [frame[0:hauteur_split,largeur_split:frame.shape[1]], 	frame[hauteur_split:frame.shape[0],largeur_split:frame.shape[1]]]
		# ]
		posX,posY = -1,-1
		for i in range(nbMClick):
			posX,posY = wich_case(mouseXY[i][0],mouseXY[i][1],frame.shape[1],frame.shape[0])	
			cv2.imwrite('../Victoire/Data/valid/normal/piste_'+str(numB)+'.png',mat_im[posX][posY])
			allPosXY.append([posX, posY])
			numB+=1
		mouseXY = []
		nbMClick = 0
		
		allPosXY = []			
		# Capture frame-by-frame
		ret, frame = cap.read()
		frame = cv2.resize(frame,(1366,768))
		
		# Display the resulting frame
		if ret :
			cv2.line(frame,(0,hauteur_split),(frame.shape[1],hauteur_split),(0,0,255),1)
			cv2.line(frame,(0,hauteur_split*2),(frame.shape[1],hauteur_split*2),(0,0,255),1)

			cv2.line(frame,(largeur_split,0),(largeur_split,frame.shape[0]),(0,0,255),1)
			cv2.line(frame,(largeur_split*2,0),(largeur_split*2,frame.shape[0]),(0,0,255),1)
			cv2.imshow('frame',frame)

		else :
			break

	if k == 27:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
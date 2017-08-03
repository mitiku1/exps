import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
def get_dataset(filename):
	print "reading file:",filename
	data=pd.read_csv(filename,sep=",",header=0)
	emotions=data.emotion
	pixels=data.pixels
	pixelList=[]
	print "Converting pixels from string to array"
	emotionList=[]

	for i in range(len(emotions)):
		p=pixels[i]
		e=emotions[i]
		ems=[0,0,0,0,0,0,0]
		ems[e]=1
		emotionList.append(ems)
		pixelList.append(np.fromstring(p,dtype=int,sep=" "))
	pixels=np.array(pixelList)
	emotions=np.array(emotionList)
	plt.imshow(np.resize(pixels[0],[48,48]))
	return pixels,emotions
def max_index(array):
	maxIndex=0;
	for i in range(len(array)):
		if(array[i]>array[maxIndex]):
			maxIndex=i
	return maxIndex
def show_image(frame,emotion):
	ems={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
	cv2.imshow(ems[max_index(emotion)], frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
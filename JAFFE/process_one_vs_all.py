import cv2
import os
from emotions import emotions
import numpy as np
import pandas as pd

def __get_emtion_start_with(start_str):
	for key in emotions:
		if(emotions[key].startswith(start_str)):
			return key
	return None


def __get_emtion_from_jaff_file(filename):
	emtion=filename.split(".")[1][:2]
	return __get_emtion_start_with(emtion)

def __get_images(images_dir):
	files=os.listdir(images_dir)
	images=[]
	emotionsList=[]
	for file in files:
		if(len(file)>4 and file[-4:]==".jpg"):
			frame=cv2.imread(images_dir+"/"+file,0)
			emotion=__get_emtion_from_jaff_file(file)
			images.append(frame)
			emotionsList.append(emotion)
	return images,emotionsList


def process_images(images_dir,_class):
	global emotions
	class_name=emotions[_class]
	images,emotionsList=__get_images(images_dir)

	output_to_file=""
	for i in range(len(images)):
		image=images[i]
		_class_output="0"
		if(_class==emotionsList[i]):
			_class_output="1"
		output_to_file+=__image_to_line(image,_class_output)+"\n"
	with open("jaffe_"+class_name+".csv","w") as f:
		f.write(output_to_file)

		
def __image_to_line(image,_class):
	# conversts the image into comma separated pixel value and appends 
	# class of the image(one hot)
	image=cv2.resize(image,(48,48))
	shape=image.shape
	image=np.reshape(image,(1,shape[0]*shape[1]))
	image_str=",".join(str(x) for x in image[0])
	image_str+=","+_class
	return image_str

def get_datasets(filename,header=None):
	data=pd.read_csv(filename,header=header)
	datasets=data.values
	return datasets[:,:-1],datasets[:,-1:]
if __name__=="__main__":
	# print get_emtion_start_with("NE")
	# process_images("jpg","jaffe_48x48.csv")

	# images,emotions=get_datasets("jaffe.csv")
	# print(images.shape,emotions.shape)
	# process_images("jpg",_class=1)
	for i in range(len(emotions)):
		process_images("jpg",_class=i)
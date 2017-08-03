from process_one_vs_all import get_datasets

from keras.models import Sequential
from keras.models import load_model
from keras.layers import *
import keras
import cv2
from emotions import emotions
import dlib
import os

def max_index(array):
	maxIndex=0;
	for i in range(len(array)):
		if(array[i]>array[maxIndex]):
			maxIndex=i
	return maxIndex
def train(_class):
	num_classes=1
	input_shape=(48,48,1)
	batch_size=50
	epochs=100
	TRAINING_DATA_SIZE=150

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(128, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# model.add(Conv2D(256, (5, 5), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='sigmoid'))
	model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(1e-6),
              metrics=['accuracy'])
	filename="jaffe_"+emotions[_class]+".csv";
	print "Loading data from",filename
	x_batch,y_batch=get_datasets("jaffe_"+emotions[_class]+".csv")
	print "Loaded data"
	x_train=np.array(x_batch[:TRAINING_DATA_SIZE]).reshape((-1,48,48,1))
	y_train=np.array(y_batch[:TRAINING_DATA_SIZE]).reshape(-1,1)
	x_test=np.array(x_batch[TRAINING_DATA_SIZE:]).reshape((-1,48,48,1))
	y_test=np.array(y_batch[TRAINING_DATA_SIZE:]).reshape(-1,1)
	model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
	output_filename="jaffe_"+emotions[_class]+"_model.h5"
	model.save(output_filename)
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

def get_image(filename,shape):
	frame=cv2.imread(filename,0)
	resized=cv2.resize(frame,shape)
	return frame,resized
def predict_from_models(models,image):
	predictions={}
	for model in models:
		p=models[model].predict(image)
		predictions.update({model:p})
	return max(predictions,key=predictions.get)

def predict(models):	
	# num_classes=1
	# input_shape=(48,48,1)

	# model = Sequential()
	# model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
	#                  activation='relu',
	#                  input_shape=input_shape))
	# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# model.add(Conv2D(128, (5, 5), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Conv2D(256, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	# # model.add(Conv2D(256, (5, 5), activation='relu'))
	# # model.add(MaxPooling2D(pool_size=(2, 2)))

	# # model.add(Conv2D(256, (5, 5), activation='relu'))
	# # model.add(MaxPooling2D(pool_size=(2, 2)))


	# model.add(Flatten())
	# model.add(Dense(1024, activation='relu'))
	# model.add(Dense(num_classes, activation='sigmoid'))
	# model.compile(loss=keras.losses.binary_crossentropy,
 #              optimizer=keras.optimizers.Adam(1e-6),
 #              metrics=['accuracy'])

	# model_filename="jaffe_"+emotions[_class]+"_model.h5"
	# print("Loading file",model_filename)
	# model.load_weights(model_filename)
	# images_folder="/home/mitiku/iCog/experments/JAFFE/jpg"
	# index=0

	# for file in os.listdir(images_folder):
	# 	index+=1
	# 	if index<120:
	# 		continue
	# 	elif(index>215):
	# 		break
	# 	filepath=images_folder+"/"+file;
	# 	original,frame=get_image(filepath,(48,48))
	# 	frame=cv2.resize(frame,(48,48))
	# 	face=np.reshape(frame,(-1,48,48,1))
	# 	emotion = predict_from_models(models,face)
	# 	# ems={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
	# 	print(emotion)
	# 	# mIndex=max_index(emotion[0])
	# 	cv2.imshow(str(emotion)+":" +file, original)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	cam = cv2.VideoCapture(0)
	ret, frame = cam.read()

	while ret:
		ret, frame = cam.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
		# rectangle=detector(frame,1)[0]
		# color=(48, 12, 160)
		# cv2.rectangle(frame, rectangle.width,rectangle.height, color)
		original=frame
		frame=cv2.resize(frame,(48,48))
		face=np.reshape(frame,(-1,48,48,1))
		# emotion = model.predict(face)
		# mIndex=max_index(emotion[0])
		# print(emotions[mIndex])
			# 	frame=cv2.resize(frame,(48,48))
		emotion = predict_from_models(models,face)
		# ems={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
		print(emotion)
		# mIndex=max_index(emotion[0])
		cv2.imshow(str(emotion)+":" , original)
		cv2.imshow("Preview", frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
	cam.release()

detector = dlib.get_frontal_face_detector()

def load_models():
	models={}
	for i in emotions:
		models[emotions[i]]=load_model("jaffe_"+emotions[i]+"_model.h5")
	return models
def main():
	TRAIN=False
	_class=0
	if(TRAIN):
		for i in emotions:
			print "Training for ",emotions[i]
			train(i)
	else:
		models=load_models()
		predict(models)
if __name__=="__main__":
	main()
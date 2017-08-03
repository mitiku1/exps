from process import get_datasets

from keras.models import Sequential
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
def train():
	num_classes=7
	input_shape=(48,48,1)
	batch_size=50
	epochs=100
	TRAINING_DATA_SIZE=130

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(9, 9), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(128, (7, 7), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# model.add(Conv2D(256, (5, 5), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
	print "Loading data from","jaffe_48x48.csv"
	x_batch,y_batch=get_datasets("jaffe_48x48.csv")
	print "Loaded data"
	x_train=np.array(x_batch[:TRAINING_DATA_SIZE]).reshape((-1,48,48,1))
	y_train=np.array(y_batch[:TRAINING_DATA_SIZE]).reshape((-1,7))
	x_test=np.array(x_batch[TRAINING_DATA_SIZE:]).reshape((-1,48,48,1))
	y_test=np.array(y_batch[TRAINING_DATA_SIZE:]).reshape((-1,7))
	model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
	model.save("fec-model_5.h5")
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

def get_image(filename,shape):
	frame=cv2.imread(filename,0)
	resized=cv2.resize(frame,shape)
	return frame,resized

def pridict():	
	num_classes=7
	input_shape=(48,48,1)

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(9, 9), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(128, (7, 7), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# model.add(Conv2D(256, (5, 5), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])



	print("Loading file")
	model.load_weights("fec-model_5.h5")
	images_folder="/home/mitiku/iCog/experments/JAFFE/jpg"
	index=0

	for file in os.listdir(images_folder):
		index+=1
		if index<120:
			continue
		filepath=images_folder+"/"+file;
		original,frame=get_image(filepath,(48,48))
		frame=cv2.resize(frame,(48,48))
		face=np.reshape(frame,(-1,48,48,1))
		emotion = model.predict(face)
		ems={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
		mIndex=max_index(emotion[0])
		cv2.imshow(ems[mIndex]+":" +file, original)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# cam = cv2.VideoCapture(0)
	# ret, frame = cam.read()

	# while ret:
	# 	ret, frame = cam.read()
	# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
	# 	# rectangle=detector(frame,1)[0]
	# 	# color=(48, 12, 160)
	# 	# cv2.rectangle(frame, rectangle.width,rectangle.height, color)

	# 	frame=cv2.resize(frame,(48,48))
	# 	face=np.reshape(frame,(-1,48,48,1))
	# 	emotion = model.predict(face)
	# 	mIndex=max_index(emotion[0])
	# 	print(emotions[mIndex])
	# 	cv2.imshow("Preview", frame)
	# 	if cv2.waitKey(10) & 0xFF == ord('q'):
	# 		break

	# cv2.destroyAllWindows()
	# cam.release()

detector = dlib.get_frontal_face_detector()

def main():
	TRAIN=False

	if(TRAIN):
		train()
	else:
		pridict()
if __name__=="__main__":
	main()
from keras.models import Sequential
from keras.layers import *
import keras
from dataprocess import get_dataset,show_image
import cv2
import emopy


def max_index(array):
	maxIndex=0;
	for i in range(len(array)):
		if(array[i]>array[maxIndex]):
			maxIndex=i
	return maxIndex

def main():
	num_classes=7
	input_shape=(48,48,1)
	batch_size=128
	epochs=25

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
	# x_batch,y_batch=get_dataset("fer2013.csv")
	# x_train=np.array(x_batch[:30000]).reshape((-1,48,48,1))
	# y_train=np.array(y_batch[:30000]).reshape((-1,7))
	# x_test=np.array(x_batch[30000:]).reshape((-1,48,48,1))
	# y_test=np.array(y_batch[30000:]).reshape((-1,7))
	# model.fit(x_train, y_train,
 #          batch_size=batch_size,
 #          epochs=epochs,
 #          verbose=1,
 #          validation_data=(x_test, y_test))
	# model.save("fec-model_4.h5")
	# score = model.evaluate(x_test, y_test, verbose=0)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])
	print("Loading file")
	model.load_weights("fec-model_4.h5")

	cam = cv2.VideoCapture(0)
	ret, frame = cam.read()

	while ret:
		ret, frame = cam.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
		frame=cv2.resize(frame,(48,48))
		face=np.reshape(frame,(-1,48,48,1))
		emotion = model.predict(face)
		ems={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
		mIndex=max_index(emotion[0])
		print(ems[mIndex])
		cv2.imshow("Preview", frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
	cam.release()


if __name__=="__main__":
	main()
	# x_batch,y_batch=get_dataset("fer2013.csv")
	# for i in range(100):
	# 	show_image(np.reshape(x_batch[i*10],(48,48)),y_batch[i])


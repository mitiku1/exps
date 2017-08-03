from process import get_datasets

from keras.models import Sequential
from keras.layers import *
import keras
import cv2

def main():	
	num_classes=7
	input_shape=(256,256,1)
	batch_size=128
	epochs=5

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])
	print "Loading data from","jaffe.csv"
	x_batch,y_batch=get_datasets("jaffe.csv")
	print "Loaded data"
	x_train=np.array(x_batch[:150]).reshape((-1,256,256,1))
	y_train=np.array(y_batch[:150]).reshape((-1,7))
	x_test=np.array(x_batch[150:]).reshape((-1,256,256,1))
	y_test=np.array(y_batch[150:]).reshape((-1,7))
	model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
	model.save("fec-model_4.h5")
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

if __name__=="__main__":
	main()
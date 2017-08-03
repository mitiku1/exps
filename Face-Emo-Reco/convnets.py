import tensorflow as tf
import numpy as np 

import cv2
from emotions import EMOTIONS

x=tf.placeholder(tf.float32,[None,768])
y_=tf.placeholder(tf.float32,[None,6])

x_image=tf.reshape(x,[-1,24,32,1])


Wc1=tf.Variable(tf.truncated_normal([5,5,1,16],stddev=0.25))
bc1=tf.Variable(tf.truncated_normal([16],stddev=.25))

conv1=tf.nn.softmax(tf.nn.conv2d(x_image,Wc1,strides=[1,1,1,1],padding="SAME")+bc1)
pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

Wc2=tf.Variable(tf.truncated_normal([5,5,16,64],stddev=.25))
bc2=tf.Variable(tf.truncated_normal([64],stddev=.25))

conv2=tf.nn.relu(tf.nn.conv2d(pool1,Wc2,strides=[1,1,1,1],padding="SAME"))
pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



Wfc1=tf.Variable(tf.truncated_normal([6*8*64,1024]))
bfc1=tf.Variable(tf.truncated_normal([1024]))

pool2_flatten=tf.reshape(pool2,[-1,6*8*64])

fc1=tf.nn.relu(tf.matmul(pool2_flatten,Wfc1)+bfc1)

keep_prob = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(fc1, keep_prob)


Wfc2=tf.Variable(tf.truncated_normal([1024,6]))
bfc2=tf.Variable(tf.truncated_normal([6]))

y_conv=tf.matmul(fc1_drop,Wfc2)+bfc2


optimizer=tf.train.GradientDescentOptimizer(0.000001);

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train=optimizer.minimize(loss)

correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y_conv,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

datasets=tf.contrib.learn.datasets.base.load_csv_without_header(

	filename="data.csv",

      target_dtype=np.int,
      features_dtype=np.float32)
images=datasets.data
labels=datasets.target;

img=cv2.imread("image.jpg",0)
print(img.shape)


def get_data(filename):
	datasets=[]
	with open(filename) as f:
		for line in f.readlines():
			line_data=[int(i) for i in line.split(",")]
			datasets.append(line_data)
	datasets=np.array(datasets)
	print("datasets shape",datasets.shape)
	return datasets[:,:-6],datasets[:,-6:]
images,labels=get_data("data.csv")

for i in range(1000):
	index=i%84
	if(index%8!=0):
		x_val=images[index]
		y_val=labels[index]
		sess.run(train,feed_dict={x:np.reshape(x_val,(1,768)),y_:np.reshape(y_val,(1,6)), keep_prob: 1})
test_images=[]
test_labels=[]
for i in range(84):
	if(i%8==0):
		test_images.append(images[i])
		test_labels.append(labels[i])
print(sess.run(accuracy,feed_dict={x:np.reshape(test_images,(-1,768)),y_:np.reshape(test_labels,(-1,6)), keep_prob: 1}))


img=cv2.resize(img,(32,24))
print(img.shape)
print("Predicted")
print(sess.run(y_conv,feed_dict={x:np.reshape(img,(-1,768)), keep_prob: 0.5}))


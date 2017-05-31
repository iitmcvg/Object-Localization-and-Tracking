import os
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import glob
from vgg16 import vgg16
totaltime=0
text_file =  open("weights.txt", "w") 
#file containing the path to images and labels
filename = './dataset.txt'
#lists where to store the paths and labels
filenames = []
labels = []

x = tf.placeholder(tf.float32, shape=[None,196608])
y_ = tf.placeholder(tf.float32, shape=[None,4])

def fclayer(x, W, b):
	fc1 = tf.add( tf.matmul(x,W), b)
	return fc1
def init_w(shape):
	return tf.Variable(tf.truncated_normal(shape,mean=0.0, stddev = 0.1),name='weights')
def init_b(shape):
	return tf.Variable(tf.constant(0.1, shape=shape),name='biases')

x_image = tf.reshape(x, [-1,256,256,3])
x_image=tf.image.resize_images(x_image,(224,224))

sess=tf.InteractiveSession()

vgg=vgg16(x_image,'vgg16_weights.npz',sess)

with tf.name_scope('fc1reg'):
	shape = int(np.prod(vgg.pool5.get_shape()[1:]))
	fcW1=init_w([shape,1024])
	fcB1=init_b([1024])
	pool5_flat = tf.reshape(vgg.pool5, [-1, shape])
	FCreg1=fclayer(pool5_flat,fcW1,fcB1)

with tf.name_scope('fc2reg'):
	fcW2=init_w([1024,512])
	fcB2=init_b([512])
	FCreg2=fclayer(FCreg1,fcW2,fcB2)

with tf.name_scope('fc3reg'):
	fcW3=init_w([512,4])
	fcB3=init_b([4])
	FCreg3=fclayer(FCreg2,fcW3,fcB3)

loss = tf.reduce_mean(tf.square((FCreg3) - (y_)))

#reading file and extracting path and labels
with open(filename, 'r') as File:
	infoFile = File.readlines()	#reading lines from files
	for line in infoFile: #reading line by line
		words = line.split(' ')
		filenames.append(words[0])
		labels.append(words[1])
		labels.append(words[2])
		labels.append(words[3])
		labels.append(words[4])

input_image = []
NumFiles = len(filenames) 

#Converting filnames into tensor
tfilenames = ops.convert_to_tensor(filenames, dtype = dtypes.string)
tlabels = ops.convert_to_tensor(labels, dtype=dtypes.string)

#creating a queue which contains the list of files to read and the values of labels
filename_queue = tf.train.slice_input_producer([tfilenames, tlabels], num_epochs=10, shuffle=False, capacity = NumFiles)

#reading image files and decoding them
rawIm = tf.read_file(filename_queue[0])
decodedIm = tf.image.decode_jpeg(rawIm)
lbl = []

#extracting the labels queue
label_queue = filename_queue[1]

#Initializing global and local variable initializers
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

#Creating an interactive session to run in python file
label_value = []
sess.run(init_op)
Tolerance = 0
no_epoch=0
loss_to_be_minimized = 0
label_counter = 0
train_step = tf.train.GradientDescentOptimizer(1e-8).minimize(loss)
Train_Checker = []
with sess.as_default():
	
	#start populating the filename queue
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	flag = 0 #epoch
	lbl_array = []
	img_array = []
	
	while(True):
		flag = flag + 1
		begtime = time.time()
		i = 0
		Train_Checker.append(loss_to_be_minimized)	#Previous loss function
		loss_to_be_minimized = 0
		for i in range(NumFiles):
			
			if flag<=1:
				nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])					
				labels =np.reshape(labels, (-1,4))
				lbl = labels[i]
				lbl_array.append(lbl)
				input_image = (sess.run(tf.reshape(image, [196608])))
				img_array.append(input_image)
				ip_img = input_image
				
			no_of_times_run = 0
			
			while(True):
				train_step.run(feed_dict={x:[img_array[i]], y_:[lbl_array[i]]})
				no_of_times_run = no_of_times_run + 1
				if no_of_times_run>3:
					break
			loss_to_be_minimized = loss_to_be_minimized +  sess.run(loss, feed_dict={x:[img_array[i]], y_:[lbl_array[i]]})
			
		endtime = time.time()
		totaltime = totaltime + (endtime-begtime)

		print ("Epoch: "+ str(flag)+ "\t"+ "Total Error: "+str(loss_to_be_minimized)+ "\t"+ "Tolerance: "+ str(Tolerance)+ "\t" + "Time Taken: "+ str(endtime- begtime))
		text_file.write(str(flag) + " "+ str(loss_to_be_minimized))
		plt.ion()
		y = loss_to_be_minimized
		plt.xlabel("Epochs")
		plt.ylabel("Total_loss(L2 Loss)")
		plt.title("Loss Vs Epochs")
		plt.scatter(flag, y)
		plt.pause(0.05)
		Train_Checker.append(loss_to_be_minimized) #Updated Loss function
		if (loss_to_be_minimized < 1000):
			break
		if (Train_Checker[0] <= Train_Checker[1] and flag > 1):
			Tolerance = Tolerance + 1
			if (Tolerance > 30):	
				break
		del Train_Checker[:]
	coord.request_stop()
	coord.join(threads)
	writer.close()
	
print("TotalTime Taken: " + str(totaltime))
for imga in glob.glob("./test/*.jpg"):
	img = cv2.imread(imga)
	clone = np.copy(img)
	prediction=np.zeros(1)
	count=0
	print ("test_image" + str(count))
	cv2.imshow('image',img )
	cv2.waitKey(0)
	img_resized = cv2.resize(img, (256,256), interpolation = cv2.INTER_LINEAR)
	image_linear = np.reshape(img_resized, [-1,196608])
	prediction= (sess.run(fc1, feed_dict={x:image_linear}))
	image_restored = np.reshape(image_linear, [256,256,3])
	count = count+1
	[[ix,iy,jx,jy]]=prediction
	print (prediction)
	cv2.rectangle(img_resized, (np.floor(ix), np.floor(iy)), (np.floor(jx), np.floor(jy)), (0,0,255), 3)
	cv2.imshow("Test_Image", img_resized)
	k = cv2.waitKey(0)

	cv2.destroyAllWindows()
text_file.close()

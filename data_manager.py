import os
import time
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import cv2
import glob
def data_load(filename):
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
	sess = tf.InteractiveSession()
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
	sess = tf.InteractiveSession()
	with sess.as_default():
		flag = 0
		lbl_array = []
		img_array = []
		while(True):
			flag = flag + 1
			i = 0
			for i in range(NumFiles):
				
				if flag<=1:
					nm, image, lb = sess.run([filename_queue[0], decodedIm, label_queue])					
					labels =np.reshape(labels, (-1,4))
					lbl = labels[i]
					lbl_array.append(lbl)
					input_image = (sess.run(tf.reshape(image, [196608])))
					img_array.append(input_image)				
	return img_array,lbl_array
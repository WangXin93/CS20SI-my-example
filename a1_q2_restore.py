import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt

# Define parameters
batch_size = 128

# Step1: Read data
mnist = input_data.read_data_sets('/tmp/data/mnist', one_hot=True) 

# Step2: Set input and labels
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.int64, [batch_size, 10], name='Y_placeholder')
keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

# Step3: Create variables
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1),
	name='w_conv1')  
b_conv1 = tf.Variable(tf.zeros([32]), name = 'b_conv1')
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1),
	name='w_conv2')
b_conv2 = tf.Variable(tf.zeros([64]), name = 'b_conv2')
w_fc1 = tf.Variable(tf.truncated_normal([7, 7, 64, 1024], stddev=0.1),
	name='w_fc1')
b_fc1 = tf.Variable(tf.zeros([1024]), name='b_fc1')
w_fc2 = tf.Variable(tf.truncated_normal([1, 1, 1024, 10], stddev=0.1),
	name='w_fc2')
b_fc2 = tf.Variable(tf.zeros([10]), name='b_fc2')

# Step4: Define model
X_reshape = tf.reshape(X, [-1,28,28,1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(X_reshape, w_conv1, [1,1,1,1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, [1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

h_fc1 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_fc1, [1,1,1,1], padding='VALID') + b_fc1)
h_dropout1 = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = tf.nn.conv2d(h_dropout1, w_fc2, [1,1,1,1], padding='VALID') + b_fc2

logits = tf.squeeze(h_fc2, [1, 2])

predictions = tf.argmax(logits, 1)

correct_preds = tf.equal(predictions, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Restore variables from disk.
	saver.restore(sess, "/tmp/lg/model.ckpt")
	print("Model have been restored.")
	
	n_batchs = mnist.test.num_examples//batch_size

	# Test accuracy from checkpoint
	total_accuracy = 0
	for i in range(n_batchs):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy,
			 feed_dict={X:X_batch, Y:Y_batch, keep_prob:1.0})
		total_accuracy += accuracy_batch
	print("Accuracy {}".format(total_accuracy/n_batchs))

	# Show the error prediction
	for i in range(3):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		# Get prediction boolean and prediction labels
		np_preds, np_pred_labels = sess.run([correct_preds, predictions],
			feed_dict={X:X_batch, Y:Y_batch, keep_prob:1.0})
		# Use boolean to select false predictions
		error_preds = [not p for p in np_preds]
		# Get the prediction error images and their origin labels
		error_images, origin_labels = X_batch[error_preds], Y_batch[error_preds]
		error_labels = np_pred_labels[error_preds]
		if sum(error_preds): # if there are error images		
			print(error_images.shape, origin_labels.shape, error_labels.shape)
			for j in range(error_labels.shape[0]):
				plt.imshow(error_images[j].reshape(28,28), cmap='gray')
				plt.title("Predict: %s\n Origin: %s" %
					(error_labels[j], np.argmax(origin_labels[j])))
				plt.axis('off')
				plt.show() 
	
	# Show accuracy for every class
	num_list = [] # list of original number
	result_list = [] # list of True and False
	for i in range(n_batchs):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		np_preds = sess.run(correct_preds,
			feed_dict={X:X_batch, Y:Y_batch, keep_prob:1.0})
		result_list.extend(list(np_preds))
		num_list.extend(list(np.argmax(Y_batch, axis=1)))
	print(len(num_list), len(result_list))
	accuracy_list = []
	num_list = np.array(num_list)
	result_list = np.array(result_list)
	for i in range(0,10):
		total_n = sum(num_list == i)
		total_right = sum(result_list[num_list==i])
		print("for num '{}', {}/{} is right".format(i, total_right, total_n))
		accuracy_list.append(total_right/total_n)
	plt.bar(np.arange(10), accuracy_list)
	plt.title("Accuracy for each class")
	plt.ylim(0.9, 1.0)
	plt.show()
	
		
	
	






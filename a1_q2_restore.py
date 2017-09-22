import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import time

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

# Step4: Define loss
X_reshape = tf.reshape(X, [-1,28,28,1])
h_conv1 = tf.nn.relu(tf.nn.conv2d(X_reshape, w_conv1, [1,1,1,1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, [1,1,1,1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

h_fc1 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_fc1, [1,1,1,1], padding='VALID') + b_fc1)
h_dropout1 = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = tf.nn.conv2d(h_dropout1, w_fc2, [1,1,1,1], padding='VALID') + b_fc2

logits = tf.squeeze(h_fc2, [1, 2])

correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# Restore variables from disk.
	saver.restore(sess, "/tmp/lg/model.ckpt")
	print("Model have been restored.")
	
	n_batchs = mnist.test.num_examples//batch_size

	total_accuracy = 0
	for i in range(n_batchs):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy,
			 feed_dict={X:X_batch, Y:Y_batch, keep_prob:1.0})
		total_accuracy += accuracy_batch
	print("Accuracy {}".format(total_accuracy/n_batchs))






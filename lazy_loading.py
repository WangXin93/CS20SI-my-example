import tensorflow as tf

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
#z = tf.add(x, y) # Add this line will become normal loading

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('/tmp/my_graph/1', sess.graph)
	for _ in range(10):
		#sess.run(z) # Now is normal loading
		sess.run(tf.add(x,y)) # Now is lazy loading
	writer.close()
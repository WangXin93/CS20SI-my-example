import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = './slr05.xls'

def huber_loss(labels, predictions, delta=1.0):
	residual = tf.abs(predictions - labels)
	def f1(): return 0.5 * tf.square(delta)
	def f2(): return delta * residual - 0.5 * tf.square(delta)
	return tf.cond(residual < delta, f1, f2)


# Step 1: read data from .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

#%%
# Step 2: create placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to zero
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Step 4: build model to infer
Y_predicted = X * w + b

# Step 5: use square loss as  loss function
#loss = tf.square(Y - Y_predicted, name='loss')
loss = huber_loss(Y, Y_predicted)

# Step 6: use GradientDescentOptimizer as optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter('/tmp/my_graph/lg', sess.graph)
      
    # turn on pyplot interactive mode
    plt.ion()
    # load origin data for plot
    np_X, np_Y = data.T[0], data.T[1]
         
    # Step 8: train the model
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
        
        # Step 9: output the values of w and b and draw every frame
        plt.cla()
        np_w, np_b = sess.run([w, b]) 
        plt.plot(np_X, np_Y, 'bo', label='Real data')
        plt.plot(np_X, np_X*np_w+np_b, 'r-', label='prediction')
        plt.legend()
  
        # plot the results
               
        plt.pause(0.001)
    plt.pause(3)  
    writer.close()

    

        
        

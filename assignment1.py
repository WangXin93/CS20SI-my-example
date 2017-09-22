"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

xb = tf.random_uniform([], minval=-1, maxval=1, name='xb')
yb = tf.random_uniform([],minval=-1, maxval=1, name='yb')
zb = tf.zeros([],name='zb')
outb = tf.case({tf.less(xb, yb):lambda:tf.add(xb, yb),
		tf.greater(xb, yb):lambda:tf.subtract(xb, yb)},
		default=lambda:zb,
		exclusive=True)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

xc = tf.constant([[0, -2, -1], [0, 1, 2]], name='xc')
yc = tf.zeros_like(xc, name='yc')
outc = tf.equal(xc, yc, name='outc')

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

xd = tf.random_uniform([20], minval=25, maxval=35, name='xd')
idxd = tf.where(tf.greater(xd, 30), name='idxd')
outd = tf.gather(xd, idxd, name='outd')

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

oute = tf.diag(tf.range(1,7), name='oute')

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

xf = tf.random_normal([10,10], name='xf')
outf = tf.matrix_determinant(xf, name='outf')

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

xg = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9], name='xg')
outg, _ = tf.unique(xg, name='outg')

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

xh = tf.random_normal([300], name='xh')
yh = tf.random_normal([300], name='yh')
outh = tf.cond(tf.less(tf.reduce_mean(xh - yh), 0),
		lambda:tf.reduce_mean(tf.square(xh-yh)),
		lambda:tf.reduce_sum(tf.abs(xh-yh)))
	

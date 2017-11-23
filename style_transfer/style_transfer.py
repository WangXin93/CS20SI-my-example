""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

import vgg_model
import utils

# parameters to manage experiments
NOISE_RATIO = 0.6 # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Layer used for content features. You can change this.
CONTENT_LAYER = 'conv4_2'

MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    return tf.reduce_sum(tf.square(p - f)) / (4.0 * p.size)


def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Hint: you'll first have to reshape F
    """
    F = tf.reshape(F, shape=[M, N])
    return tf.matmul(tf.transpose(F), F)


def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    N = a.shape[3] # number of filters
    M = a.shape[1] * a.shape[2] # height times width of the feature map
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum(tf.square(G - A) / ((2 * N * M) ** 2))

def _create_style_loss(A, model, W):
    """ Return the total style loss
    Args:
        W: Style loss weights of five layers, give more weights to deeper layers.
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    ###############################
    ## TO DO: return total style loss
    return sum([W[i]*E[i] for i in range(n_layers)])

    ###############################

def _create_losses(model, input_image, content_image, style_image, W, CONTENT_WEIGHT, STYLE_WEIGHT):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model, W)

        ##########################################
        ## TO DO: create total loss. 
        ## Hint: don't forget the content loss and style loss weights
        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
        ##########################################

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.name_scope('summaries'):
        tf.summary.scalar('content loss', model['content_loss'])
        tf.summary.scalar('style loss', model['style_loss'])
        tf.summary.scalar('total loss', model['total_loss'])
        tf.summary.histogram('histogram content loss', model['content_loss'])
        tf.summary.histogram('histogram style loss', model['style_loss'])
        tf.summary.histogram('histogram total loss', model['total_loss'])
        return tf.summary.merge_all()

def train(model, generated_image, initial_image, dt, ITERS, SAVE_EVERY):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.

    Args:
        dt: subdirectory's name, in a datetime format
    """
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ###############################
        ## TO DO: 
        ## 1. initialize your variables
        ## 2. create writer to write your graph
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('graphs', sess.graph)
        ###############################
        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < SAVE_EVERY:
                skip_step = 10
            elif index >= SAVE_EVERY:
                skip_step = SAVE_EVERY
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TO DO: obtain generated image and loss
                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], model['summary_op']])

                ###############################
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = 'outputs/%s/%d.png' % (dt, index)
                utils.save_image(filename, gen_image)

                #if (index + 1) % SAVE_EVERY == 0:
                #   saver.save(sess, 'checkpoints/%s/style_transfer' % dt, index)

def main(args):
    with tf.variable_scope('input') as scope:
        # use variable instead of placeholder because we're training the intial image to make it
        # look like both the content image and the style image
        input_image = tf.Variable(np.zeros([1, args.image_height, args.image_width, 3]), dtype=tf.float32)
    
    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    utils.make_dir('checkpoints')
    utils.make_dir('outputs')
    dt = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    utils.make_dir('outputs/%s' % dt)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    content_image = utils.get_resized_image(args.content, args.image_height, args.image_width)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(args.style, args.image_height, args.image_width)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                   input_image, content_image, style_image, 
                                            eval(args.W), args.content_weight, args.style_weight)
    ###############################
    ## TO DO: create optimizer
    ## model['optimizer'] = ...
    model['optimizer'] = tf.train.AdamOptimizer(args.lr).minimize(model['total_loss'], global_step=model['global_step'])
    ###############################
    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, args.image_height, args.image_width, NOISE_RATIO)
    train(model, input_image, initial_image, dt, args.iters, args.save_every)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--style', type=str,
	help='Style want to transfer to another image.', default='styles/guernica.jpg')
    parser.add_argument('--content', type=str,
        help='Content wiil be combined with style.', default='content/deadpool.jpg')
    parser.add_argument('--image_width', type=int,
        help='Width of image to be generated.', default=333)
    parser.add_argument('--image_height', type=int,
        help='Height of image to be generated.', default=250)
    parser.add_argument('--W', type=str,
        help='Style loss weights of five layers, give more weights to deeper layers.', default=[0.5, 1.0, 1.5, 3.0, 4.0])
    parser.add_argument('--content_weight', type=float,
        help='Weight of content loss on total loss.', default=0.01)
    parser.add_argument('--style_weight', type=float,
        help='Wight of style loss on total loss.', default=1.0)
    parser.add_argument('--lr', type=float,
        help='Learning rate of training.', default=2.0)
    parser.add_argument('--iters', type=int,
        help='Number of iterations during training.', default=300)
    parser.add_argument('--save_every', type=int,
        help='Number of steps between saving images.', default=20) 

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

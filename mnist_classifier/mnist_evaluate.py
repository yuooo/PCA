from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import mnist_classifier as mc

FLAGS = None
       
def main(_):
    with tf.Graph().as_default() as g:
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Build the graph for the deep net
        y_conv, keep_prob = mc.deepnn(x)


        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)

        # Create adversarial images with FSG Method

        img_gradients = tf.gradients(cross_entropy, x)[0]
        epsilon = 0.2
        adversarial_images = x + epsilon*tf.sign(img_gradients)


        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver()

        with tf.Session(config = config) as sess:

            checkpoint_dir = './models'
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

            print('train accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.train.images[1:10000], y_: mnist.train.labels[1:10000], keep_prob: 1.0}))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.test.images[1:10000], y_: mnist.test.labels[1:10000], keep_prob: 1.0}))
            

#            print(train.shape)
#            print(train)
#            print('test...')
#            print(mnist.test.labels[0:20])

            adv_images = adversarial_images.eval(feed_dict = {
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 

            # # gradients = img_gradients.eval(feed_dict = {
            # #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print('adversarial accuracy %g' % accuracy.eval(feed_dict={
                    x: adv_images, y_: mnist.test.labels, keep_prob: 1.0}))

            # print(gradients[0])
            # print(adv_images[0])
            
            




    


if __name__ == '__main__':
  pca_first_class()
  print()
  print()
  print()
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

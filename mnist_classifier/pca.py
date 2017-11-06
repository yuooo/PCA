# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 13:48:12 2017

@author: jhh2677
"""

#%% GET TRAINING REPRESENTATION

#%% PCA BY CLASS ON TRAIN

#%% GET TEST AND AVERSARIAL REPRESENTATION

#%% PROJECT TEST ON LOW-EIGENSPACE OF TRAIN

#%% PLOT DIST TO THE MEAN, BY CLASS, TEST/ADV

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
import matplotlib

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import mnist_classifier as mc

FLAGS = None
    
def low_eigenvalues_projection(pca, X, n_eigenvalues):
#    print("start low_eigenvalues_projection")
    check_is_fitted(pca, ['mean_', 'components_'], all_or_any=all)

    X = check_array(X)
    if pca.mean_ is not None:
        X = X - pca.mean_
    X_transformed = np.dot(X, pca.components_[-n_eigenvalues:].T)
    if pca.whiten:
        X_transformed /= np.sqrt(pca.explained_variance_)
#    print("end low_eigenvalues_projection")
    print()
    return X_transformed

def get_dist(X):
    return np.sum(X**2, axis=1)

def adversarial_classification(X, y, class0, pca, n_eigen, name):
#    print("start adversarial_classification")
    
    index_class0 = y == class0
    X_for_PCA = X[index_class0]
    
    X_shrunk = low_eigenvalues_projection(pca, X_for_PCA, n_eigen)
    X_norm = get_dist(X_shrunk)
    print("%s average norm %g" % (name, np.mean(X_norm)))
    
    adv_predicted = np.sum(X_norm > 0.01) 
    
    print("%s classified as adversary: %d, or %g percents.\n" % (name, adv_predicted, adv_predicted*100/ len(X_norm)))
#    print("end adversarial_classification")
    
def pca_first_class(train, test, adv, noise, y_train, y_test, y_adv, y_noise, y_real):
    print("start pca_first_class")
    ### PCA on train for first class
    class0 = y_train[0]
    index_train_class0 = y_train == class0
    train_for_PCA = train[index_train_class0]
    
    pca = PCA()
    pca.fit(train_for_PCA)
    n_eigenvalues_kept = 10
    
    adversarial_classification(test, y_test, class0, pca, n_eigenvalues_kept, "test")
    adversarial_classification(adv, y_adv, class0, pca, n_eigenvalues_kept, "adversary")
    adversarial_classification(noise, y_noise, class0, pca, n_eigenvalues_kept, "noise")
    print("\nend pca_first_class \n\n\n")
    
    

def classifier(_):
    print("start main")
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
        
        predicted_labels = tf.argmax(y_conv, 1)
        real_labels = tf.argmax(y_, 1)

        # Create adversarial images with FSG Method
        img_gradients = tf.gradients(cross_entropy, x)[0]
        epsilon = 0.2
        adversarial_images = x + epsilon*tf.sign(img_gradients)

        # Create noisy data
        noisy_image = x + epsilon*tf.random_uniform(tf.shape(x), minval=-1, maxval=1, seed=12)
        
        # Others
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
            
            y_test = predicted_labels.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            
            adv_images = adversarial_images.eval(feed_dict = {
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 
            
            y_adv = predicted_labels.eval(feed_dict={
                    x: adv_images, y_: mnist.test.labels, keep_prob: 1.0})
            
            y_real = real_labels.eval(feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
            noise_images = noisy_image.eval(feed_dict = {
                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
            y_noise = predicted_labels.eval(feed_dict={
                    x: noise_images, y_: mnist.test.labels, keep_prob: 1.0})
    
            print('train accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.train.images[1:10000], y_: mnist.train.labels[1:10000], keep_prob: 1.0}))
            print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.test.images[1:10000], y_: mnist.test.labels[1:10000], keep_prob: 1.0}))
            print('adversarial accuracy %g' % accuracy.eval(feed_dict={
                    x: adv_images, y_: mnist.test.labels, keep_prob: 1.0}))
            print('noise accuracy %g' % accuracy.eval(feed_dict={
                    x: noise_images, y_: mnist.test.labels, keep_prob: 1.0}))

    
            
            
            print("end main")
            print()
            return y_test, adv_images, y_adv, y_real, noise_images, y_noise
            
            
#            print('train accuracy %g' % accuracy.eval(feed_dict={
#                    x: mnist.train.images[1:10000], y_: mnist.train.labels[1:10000], keep_prob: 1.0}))
#            print('test accuracy %g' % accuracy.eval(feed_dict={
#                    x: mnist.test.images[1:10000], y_: mnist.test.labels[1:10000], keep_prob: 1.0}))
#            print('adversarial accuracy %g' % accuracy.eval(feed_dict={
#                    x: adv_images, y_: mnist.test.labels, keep_prob: 1.0}))
#            print('noise accuracy %g' % accuracy.eval(feed_dict={
#                    x: noise_images, y_: mnist.test.labels, keep_prob: 1.0}))

#            

#            print(train.shape)
#            print(train)

#            adv_images = adversarial_images.eval(feed_dict = {
#                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 

            # # gradients = img_gradients.eval(feed_dict = {
            # #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
           
            # print(gradients[0])
            # print(adv_images[0])


if __name__ == '__main__':
    # Get the predicted labels for test and aversarial
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    y_test, adv_images, y_adv, y_real, noise_images, y_noise = classifier([sys.argv[0]] + unparsed)
    print()
    print()
    
    # PCA
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=False)
    train = mnist.train.images
    y_train = mnist.train.labels
    test = mnist.test.images
    
    pca_first_class(train, test, noise_images, adv_images, y_train, y_test, y_adv, y_noise, y_real)
    
    
#    print(y_test[:10])
#    print(y_real[:10])
#    print(y_adv[:10])

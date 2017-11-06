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
    print("start low_eigenvalues_projection")
    check_is_fitted(pca, ['mean_', 'components_'], all_or_any=all)

    X = check_array(X)
    if pca.mean_ is not None:
        X = X - pca.mean_
    X_transformed = np.dot(X, pca.components_[-n_eigenvalues:].T)
    if pca.whiten:
        X_transformed /= np.sqrt(pca.explained_variance_)
    print("end low_eigenvalues_projection")
    print()
    return X_transformed



def get_dist(X, mu):
    return np.sum((X-mu)**2, axis=1)

def pca_first_class(train, test, adv, y_train, y_test, y_adv, y_real):
    print("start pca_first_class")
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=False)
    
    
    ### PCA on train for first class
    label_for_PCA = y_train[0]
    index_train_class_0 = y_train == label_for_PCA
    train_for_PCA = train[index_train_class_0]
    
    index_test_class_0 = y_test == label_for_PCA
    test_for_PCA = test[index_test_class_0]
    
    index_adv_class_0 = y_adv == label_for_PCA
    adv_for_PCA = adv[index_adv_class_0]
        
#    print(train_for_PCA.shape)
#    print(mnist.train.labels[index_class_0])

    pca = PCA()
    pca.fit(train_for_PCA)
#    print(pca)
    
#    print(pca.explained_variance_ratio_[:10])
#    print(pca.explained_variance_ratio_[-10:])
    n_eigenvalues_kept = 10
    train_shrunk = low_eigenvalues_projection(pca, train_for_PCA, n_eigenvalues_kept)
#    print(train_shrunk[:10])
    train_norm = np.linalg.norm(train_shrunk, axis=1)
    print("train_norm average %g" % np.mean(train_norm))

    test_shrunk = low_eigenvalues_projection(pca, test_for_PCA, n_eigenvalues_kept)
#    print(test_shrunk[:10])
    test_norm = np.linalg.norm(test_shrunk, axis=1)
    print("test_norm average %g" % np.mean(test_norm))
    
    adv_shrunk = low_eigenvalues_projection(pca, adv_for_PCA, n_eigenvalues_kept)
#    print(adv_shrunk[:10])
    adv_norm = np.linalg.norm(adv_shrunk, axis=1)
    print("adv_norm average %g" % np.mean(adv_norm))
    
#    x = range(len(train_norm))
#    plt.plot(train_norm, marker="*")
#    plt.show()

    ### Classify as adversarial or not
    adv_test_predicted = np.sum(test_norm > 0.01) 
    adv_adv_predicted = np.sum(adv_norm > 0.01) 
    
    print("test classified as adversary:")
    print(adv_test_predicted)
    print(adv_test_predicted / len(test_norm))
    print()
    
    print("adversary classified as adversary:")
    print(adv_adv_predicted)
    print(adv_adv_predicted / len(adv_norm))
    print()
    
    print("end pca_first_class")
    print()
    
    

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
            
            print("end main")
            print()
            return y_test, adv_images, y_adv, y_real
            
            
#            print('train accuracy %g' % accuracy.eval(feed_dict={
#                    x: mnist.train.images[1:10000], y_: mnist.train.labels[1:10000], keep_prob: 1.0}))
#            print('test accuracy %g' % accuracy.eval(feed_dict={
#                    x: mnist.test.images[1:10000], y_: mnist.test.labels[1:10000], keep_prob: 1.0}))
#            

#            print(train.shape)
#            print(train)

#            adv_images = adversarial_images.eval(feed_dict = {
#                    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 

            # # gradients = img_gradients.eval(feed_dict = {
            # #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#            print('adversarial accuracy %g' % accuracy.eval(feed_dict={
#                    x: adv_images, y_: mnist.test.labels, keep_prob: 1.0}))

            # print(gradients[0])
            # print(adv_images[0])


if __name__ == '__main__':
    # Get the predicted labels for test and aversarial
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    y_test, adv_images, y_adv, y_real = classifier([sys.argv[0]] + unparsed)
    print()
    print()
    
    # PCA
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=False)
    train = mnist.train.images
    y_train = mnist.train.labels
    test = mnist.test.images
    
    pca_first_class(train, test, adv_images, y_train, y_test, y_adv, y_real)
    
    print()
    print()
    print()
    
#    print(y_test[:10])
#    print(y_real[:10])
#    print(y_adv[:10])

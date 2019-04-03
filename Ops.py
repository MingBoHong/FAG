import tensorflow as tf

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
import config
import numpy as np


def conv_layer(input, filter, kernel, stride=2, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network

def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Leaky_Relu(x):
    return tf.nn.leaky_relu(x)

def Deconv2d(input,filter,kernel,strides=2,layer_name="Deconv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d_transpose(inputs=input,filters=filter,kernel_size=kernel,
                                             strides=strides,padding="SAME",use_bias=False)

        return network


def Conv2d_BN_leaky(input, filter, kernel, is_training,scope,stride=2,
                    layer_name="Conv2d_BN_leaky"):
    network = conv_layer(input, filter, kernel, stride=2, layer_name="conv")
    network = Batch_Normalization(network, is_training, scope=scope + '_BN')
    network = Leaky_Relu(network)
    return network

def Conv2d_leaky(input, filter, kernel,stride=2, layer_name="Conv2d_BN"):
    network = conv_layer(input, filter, kernel, stride=2, layer_name="Conv")
    network = Leaky_Relu(network)
    return network

def Deconv2d_bn_leaky(input,filter,kernel,scope,is_training,strides=2):
    network = Deconv2d(input=input,filter=filter,kernel=kernel,strides=2)
    network = Batch_Normalization(network, is_training, scope=scope + '_BN')
    network = Leaky_Relu(network)
    return network
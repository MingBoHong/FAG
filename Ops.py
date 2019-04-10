import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

batch = 16

def conv_layer(input, filter, kernel, stride, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride)
        return network

def dropout(x, rate, training):
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

def Deconv2d(input,filter,kernel, out_shape, input_channel,strides,layer_name):
    with tf.name_scope(layer_name):
        #network = tf.layers.conv2d_transpose(inputs=input, filters=filter, kernel_size=kernel,strides=strides, padding="SAME", use_bias=False)
        kernel = tf.get_variable(layer_name+"kernel_weights", shape=[kernel, kernel, filter,input_channel], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                 dtype=tf.float32)
        network = tf.nn.conv2d_transpose(value=input, filter=kernel, strides=[1, strides, strides,1], output_shape=out_shape)

        return network


def Conv2d_BN_leaky(input, filter, kernel, is_training,scope,stride,pad=[0, 0]):
    input = tf.pad(input,[[0,0],pad,pad,[0,0]])
    network = conv_layer(input, filter, kernel, stride, layer_name=scope +"conv")
    network = Batch_Normalization(network, is_training, scope=scope + '_BN')
    network = Leaky_Relu(network)
    return network

def Conv2d_leaky(input, filter, kernel,scope,stride):
    network = conv_layer(input, filter, kernel, stride=stride, layer_name=scope + "conv")
    network = Leaky_Relu(network)
    return network

def Deconv2d_bn_leaky(input, filter, kernel, scope, out_shape, is_training,input_channel,strides):
    network = Deconv2d(input=input, filter=filter, kernel=kernel,input_channel=input_channel, strides=strides, layer_name=scope + "Dcon2D",out_shape=out_shape)
    network = Batch_Normalization(network, is_training, scope=scope + '_BN')
    network = Leaky_Relu(network)
    return network


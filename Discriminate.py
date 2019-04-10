import tensorflow as tf
from Ops import dropout, Conv2d_leaky, Conv2d_BN_leaky

def age_D(data, age_label, is_training):

    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        net1 = tf.concat([data, age_label],axis=3)
        net1 = tf.pad(net1, [[0, 0], [1, 1], [1, 1], [0, 0]])
        net1 = Conv2d_leaky(input=net1, filter=32, kernel=[3, 3], stride=1, scope='Dnet1')
        net2 = Conv2d_BN_leaky(input=net1, filter=64, kernel=[3, 3], is_training=is_training, stride=2
                               , scope='Dnet2', pad=[1, 1])
        net3 = Conv2d_BN_leaky(input=net2, filter=64, kernel=[3, 3], is_training=is_training, stride=1
                               , scope='Dnet3', pad=[1, 1])
        net4 = Conv2d_BN_leaky(input=net3, filter=128, kernel=[3, 3], is_training=is_training, stride=2
                               , scope='Dnet4', pad=[1, 1])
        net5 = Conv2d_BN_leaky(input=net4, filter=128, kernel=[3, 3], is_training=is_training, stride=1
                               , scope='Dnet5', pad=[1, 1])
        net6 = Conv2d_BN_leaky(input=net5, filter=256, kernel=[3, 3], is_training=is_training, stride=2
                               , scope='Dnet6', pad=[1, 1])
        net7 = Conv2d_BN_leaky(input=net6, filter=256, kernel=[3, 3], is_training=is_training, stride=1
                               , scope='Dnet7', pad=[1, 1])
        net8 = Conv2d_BN_leaky(input=net7, filter=512, kernel=[3, 3], is_training=is_training, stride=2
                               , scope='Dnet8', pad=[1, 1])
        net9 = Conv2d_BN_leaky(input=net8, filter=512, kernel=[3, 3], is_training=is_training, stride=1
                               , scope='Dnet9', pad=[1, 1])
        net10 = Conv2d_BN_leaky(input=net9, filter=512, kernel=[3, 3], is_training=is_training, stride=1
                               , scope='Dnet10', pad=[1, 1])

        net11 = tf.layers.flatten(net10)

        net11 = dropout(net11, 0.3, is_training)

        net11 = tf.layers.dense(inputs=net11, units=256)

        net11 = dropout(net11, 0.3, is_training)

        net11 = tf.layers.flatten(net11)

        net12 = tf.layers.dense(inputs=net11, units=1, activation='sigmoid')


    return net12
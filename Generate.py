import tensorflow as tf
from Ops import Conv2d_leaky,Deconv2d_bn_leaky,Conv2d_BN_leaky
import Train
def age_G(data, is_training, age_label, image_shape = [112, 112, 3]):

    """

    :param data:
    :param is_training:
    :param data_shape: Attention  IN tf, 112 112 3!!
    :return:
    """
    with tf.variable_scope('Generate'):
        data = tf.pad(data, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='constant')
        net1 = Conv2d_leaky(data, filter=32, kernel=[9, 9], stride=1, layer_name="Glayer1_con2d_leaky", scope='Gnet1')
        net2 = Conv2d_BN_leaky(net1, filter=64, kernel=[3, 3], stride=2, is_training=is_training,
                               layer_name="Glayer2_con2d_bn_leaky", scope='Gnet2', pad=[1, 1])
        net3 = Conv2d_BN_leaky(net2, filter=128, kernel=[3, 3], stride=2, is_training=is_training,
                               layer_name="Glayer3_con2d_bn_leaky", scope='Gnet3', pad=[1, 1])
        net4 = Conv2d_BN_leaky(net3, filter=256, kernel=[3, 3], stride=2, is_training=is_training,
                               layer_name="Glayer4_con2d_bn_leaky", scope='Gnet4', pad=[1, 1])
        net5 = Conv2d_BN_leaky(net4, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                               layer_name="Glayer5_con2d_bn_leaky", scope='Gnet5', pad=[1, 1])
        net6 = Conv2d_BN_leaky(net5, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                               layer_name="Glayer6_con2d_bn_leaky", scope='Gnet6', pad=[1, 1])
        net7 = Conv2d_BN_leaky(net6, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                               layer_name="Glayer7_con2d_bn_leaky", scope='Gnet7', pad=[1, 1])
        net8 = Conv2d_BN_leaky(net7, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                               layer_name="Glayer8_con2d_bn_leaky", scope='Gnet8', pad=[1, 1])

        net8 = tf.concat([net8, age_label], axis=3)



        # #Dnet1 = Deconv2d_bn_leaky(net8, filter=196, strides=2, kernel=3, is_training=is_training, scope='GDnet1',
        #                           out_shape=[Train.batch, int(image_shape[0]/4), int(image_shape[0]/4), 196], input_channel=net8.get_shape()[-1])
        Dnet2 = Deconv2d_bn_leaky(net8, filter=128, strides=2, kernel=3, is_training=is_training, scope='GDnet2',
                                  out_shape=[Train.batch, int(image_shape[0]/4), int(image_shape[0]/4), 128], input_channel=net8.get_shape()[-1])

        Dnet2 = tf.concat([Dnet2, net3], axis=3)


        # Dnet3 = Deconv2d_bn_leaky(Dnet2, filter=128, strides=2, kernel=3, is_training=is_training, scope='GDnet3',
        #                           out_shape=[Train.batch, int(image_shape[0]/2), int(image_shape[0]/2), 128], input_channel=Dnet2.get_shape()[-1])
        Dnet4 = Deconv2d_bn_leaky(Dnet2, filter=64, strides=2, kernel=3, is_training=is_training, scope='GDnet4',
                                  out_shape=[Train.batch, int(image_shape[0]/2), int(image_shape[0]/2), 64], input_channel=Dnet2.get_shape()[-1])

        Dnet4 = tf.concat([Dnet4, net2], axis=3)

        # Dnet5 = Deconv2d_bn_leaky(Dnet4, filter=64, strides=2, kernel=3, is_training=is_training, scope='GDnet5',
        #                           out_shape=[Train.batch, int(image_shape[0]), int(image_shape[0]), 64], input_channel=Dnet4.get_shape()[-1])
        Dnet6 = Deconv2d_bn_leaky(Dnet4, filter=32, strides=2, kernel=3, is_training=is_training, scope='GDnet6',
                                  out_shape=[Train.batch, int(image_shape[0]), int(image_shape[0]), 32], input_channel=Dnet4.get_shape()[-1])

        Dnet6 = tf.concat([Dnet6, net1], axis=3)

        # Dnet7 = Deconv2d_bn_leaky(Dnet6, filter=32, strides=1, kernel=3, is_training=is_training, scope='GDnet7',
        #                           out_shape=[Train.batch, image_shape[0], image_shape[0], 32], input_channel=Dnet6.get_shape()[-1])
        Dnet8 = Deconv2d_bn_leaky(Dnet6, filter=3, strides=1, kernel=3, is_training=is_training, scope='GDnet8',
                                  out_shape=[Train.batch, image_shape[0], image_shape[0], 3], input_channel=Dnet6.get_shape()[-1])

        net = Dnet8
        net = tf.nn.tanh(net)
        #net = tf.nn.softmax(Dnet8)#???

    return net
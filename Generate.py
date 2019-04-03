import tensorflow as tf
import config
from Ops import Drop_out,Batch_Normalization,Relu,Conv2d_leaky,Deconv2d_bn_leaky,Conv2d_BN_leaky,Deconv2d

def age_G(data,is_training,data_shape=[3,112,112]):

    """

    :param data:
    :param is_training:
    :param data_shape: Attention  IN tf, 112 112 3!!
    :return:
    """
    net1 = Conv2d_leaky(data,filter=32,kernel=[9,9],stride=1,layer_name="Glayer1_con2d_leaky")
    net2 = Conv2d_BN_leaky(net1,filter=64,kernel=[3,3],stride=2,is_training=is_training,
                           layer_name="Glayer2_con2d_bn_leaky")
    net3 = Conv2d_BN_leaky(net2, filter=128, kernel=[3, 3], stride=2, is_training=is_training,
                           layer_name="Glayer3_con2d_bn_leaky")
    net4 = Conv2d_BN_leaky(net3, filter=256, kernel=[3, 3], stride=2, is_training=is_training,
                           layer_name="Glayer4_con2d_bn_leaky")
    net5 = Conv2d_BN_leaky(net4, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                           layer_name="Glayer5_con2d_bn_leaky")
    net6 = Conv2d_BN_leaky(net5, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                           layer_name="Glayer6_con2d_bn_leaky")
    net7 = Conv2d_BN_leaky(net6, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                           layer_name="Glayer7_con2d_bn_leaky")
    net8 = Conv2d_BN_leaky(net7, filter=256, kernel=[3, 3], stride=1, is_training=is_training,
                           layer_name="Glayer8_con2d_bn_leaky")
    #??? concat?
    Dnet1 = Deconv2d_bn_leaky(net8,filter=196, strides=2, kernel=[3,3],is_training=is_training)
    Dnet2 = Deconv2d_bn_leaky(Dnet1, filter=128, strides=2, kernel=[3, 3], is_training=is_training)

    # ??? concat?

    Dnet3 = Deconv2d_bn_leaky(Dnet2, filter=128, strides=2, kernel=[3, 3], is_training=is_training)
    Dnet4 = Deconv2d_bn_leaky(Dnet3, filter=64, strides=2, kernel=[3, 3], is_training=is_training)

    # ??? concat?
    Dnet3 = Deconv2d_bn_leaky(Dnet2, filter=64, strides=2, kernel=[3, 3], is_training=is_training)
    Dnet4 = Deconv2d_bn_leaky(Dnet3, filter=32, strides=2, kernel=[3, 3], is_training=is_training)

    # ??? concat?
    Dnet5 = Deconv2d_bn_leaky(Dnet2, filter=32, strides=1, kernel=[3, 3], is_training=is_training)
    Dnet6 = Deconv2d_bn_leaky(Dnet3, filter=3, strides=1, kernel=[3, 3], is_training=is_training)

    # more?..
import tensorflow as tf
from Ops import Drop_out,Batch_Normalization,Relu,Conv2d_leaky,Deconv2d_bn_leaky,Conv2d_BN_leaky,Deconv2d

def age_D(data,age_label,data_shape=[3,112,112],is_training):


    #???concat

    net1 = Conv2d_leaky(input=data,filter=32,kernel=[3,3],stride=1,layer_name="Dlayer1_conv2d_leaky")
    net2 = Conv2d_BN_leaky(input=net1,filter=64,kernel=[3,3],is_training=is_training,stride=2
                           )
    net3 = Conv2d_BN_leaky(input=net2, filter=64, kernel=[3, 3], is_training=is_training,stride=1
                           )
    net4 = Conv2d_BN_leaky(input=net3, filter=128, kernel=[3, 3], is_training=is_training,stride=2
                           )
    net5 = Conv2d_BN_leaky(input=net4, filter=128, kernel=[3, 3], is_training=is_training,stride=1
                           )
    net6 = Conv2d_BN_leaky(input=net5, filter=256, kernel=[3, 3], is_training=is_training,stride=2
                           )
    net7 = Conv2d_BN_leaky(input=net6, filter=256, kernel=[3, 3], is_training=is_training,stride=1
                           )
    net8 = Conv2d_BN_leaky(input=net7, filter=512, kernel=[3, 3], is_training=is_training,stride=2
                           )
    net9 = Conv2d_BN_leaky(input=net8, filter=512, kernel=[3, 3], is_training=is_training,stride=1
                           )
    net10 = Conv2d_BN_leaky(input=net9, filter=512, kernel=[3, 3], is_training=is_training,stride=1
                           )
    return net10
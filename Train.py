import os
import Discriminate
import Generate
import Data_preprocess
import numpy as np
import tensorflow as tf
checkpoint_path = 'model/'
event_log_path = 'event-log'
Max_step = 60000
batch = 16
input_shape = [None, 112, 112, 3]
Gage_shape = [None, 14, 14, 4]
Dage_shape = [None, 112, 112, 4]
Lr_G = 2e-4
Lr_D = 0.01
from scipy import misc

def produce_label(shape,label_list):
    label = []
    for index in range(batch):
        temp = np.zeros([shape[1], shape[2], shape[3]])
        classes = int(label_list[index])
        temp[:, :, classes] = 1
        label.append(temp)

    return np.array(label).astype(float)

def train():

    is_training = tf.cast(True, tf.bool)
    #is_testing = tf.cast(False, tf.bool)

    with tf.variable_scope('age_label'):
        G_label = tf.placeholder(shape=Gage_shape, dtype=tf.float32, name="G_label")
        D_label = tf.placeholder(shape=Dage_shape, dtype=tf.float32, name="D_label")



    with tf.variable_scope('real_image'):
        G_image = tf.placeholder(shape=input_shape, dtype=tf.float32, name="D_in")
        D_image = tf.placeholder(shape=input_shape, dtype=tf.float32, name="D_in")

    with tf.variable_scope('G_OUT'):
        G_fake = Generate.age_G(data=G_image, age_label=G_label, is_training=is_training) + G_image


    with tf.variable_scope('D_OUT'):
        D_Tprob = Discriminate.age_D(data=D_image, age_label=D_label, is_training=is_training)

        D_Fprob = Discriminate.age_D(data=G_fake, age_label=D_label, is_training=is_training)


    with tf.variable_scope('Loss'):

        Euc_loss = (tf.reduce_mean(tf.square(G_image-G_fake)))
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Tprob, labels=tf.ones_like(D_Tprob))
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Fprob, labels=tf.zeros_like(D_Fprob))
        D_loss = tf.reduce_mean(d_loss_real+d_loss_fake)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Fprob, labels=tf.ones_like(D_Fprob)))+Euc_loss

    g_var = [var for var in tf.global_variables() if 'Generate' in var.name]
    d_var = [var for var in tf.global_variables() if 'Discriminator' in var.name]


    D_train = tf.train.MomentumOptimizer(learning_rate=Lr_D,momentum=0.9).minimize(D_loss, var_list=d_var)
    G_train = tf.train.AdamOptimizer(learning_rate=Lr_G, beta1=0.5).minimize(G_loss, var_list=g_var)

    saver = tf.train.Saver(var_list=tf.global_variables())

    data_generate = Data_preprocess.Image_dataGenerator(mode='train', batch=batch)
    tf.summary.image(tensor=G_image, name='G_real_image')
    tf.summary.image(tensor=G_fake, name='fake_image')
    tf.summary.image(tensor=D_image, name='D_real_image')
    tf.summary.scalar(tensor=G_loss, name='G_loss')
    tf.summary.scalar(tensor=D_loss, name='D_loss')
    for s in range(Max_step):

        gen_data, dis_data = data_generate.get_next_batch()

        gen_input = gen_data[:][0]
        #gen_label = gen_data[:][1]


        dis_input = dis_data[:][0]
        dis_label = dis_data[:][1]


        gin_label = produce_label(Gage_shape, dis_label)
        dis_label = produce_label(Dage_shape, dis_label)



        with tf.Session() as sess:
            all_summary_obj = tf.summary.merge_all()
            Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
            sess.run(tf.global_variables_initializer())
            _, D_loss_value,fake_image= sess.run([D_train, D_loss,G_fake], feed_dict={D_image: dis_input, D_label: dis_label, G_image: gen_input, G_label: gin_label})
            _, G_loss_value,all_summaries = sess.run([G_train, G_loss,all_summary_obj], feed_dict={D_image: dis_input, D_label: dis_label, G_image: gen_input, G_label: gin_label})

            print("Generator Loss: {}".format(G_loss_value))
            print("Discriminate Loss: {}".format(D_loss_value))


            if s%10==0:
                Event_writer.add_summary(summary=all_summaries, global_step=s)
                variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')
                saver.save(sess, variables_save_path,
                            global_step=s)



if __name__ == '__main__':
    train()
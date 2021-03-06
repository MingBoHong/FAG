import os
import Discriminate
import Generate
from Data_process_v2 import *
import numpy as np
import tensorflow as tf
from Ops import Save_batch_IMG
checkpoint_path = 'model/'
event_log_path = 'event-log'
IMG_path = 'IMG/'
Test_imgpath='Result/'
Max_step = 60000
batch = 16
input_shape = [None, 112, 112, 3]
Gage_shape = [None, 14, 14, 4]
Dage_shape = [None, 112, 112, 4]
Lr_G = 1e-4
Lr_D = 1e-4
decay_step = [2000, 4000, 6000, 4000, 70000]



def scale_img(img):
 img = img[:, :, [0, 1, 2]]
 img = np.transpose(img, [0, 2, 1])
 img[img > 1] = 1
 img[img < -1] = -1
 return ((img + 1.0) * 127.5).astype(np.uint8)


def produce_label(shape, label_list, batch):
    label = []
    for index in range(batch):
        temp = np.zeros([shape[1], shape[2], shape[3]])
        classes = int(label_list[index])
        temp[:, :, classes] = 1
        label.append(temp)

    return np.array(label).astype(float)





def train():
        global_step = tf.Variable(initial_value=0, trainable=False)
        with tf.variable_scope('age_label'):
            G_label = tf.placeholder(shape=Gage_shape, dtype=tf.float32, name="G_label")
            D_label = tf.placeholder(shape=Dage_shape, dtype=tf.float32, name="D_label")


        with tf.variable_scope('is_traing'):
            is_training = tf.placeholder(dtype=tf.bool)
        with tf.variable_scope('real_image'):
            G_image = tf.placeholder(shape=input_shape, dtype=tf.float32, name="G_in")
            tf.summary.image('Gin_image', G_image)
            D_image = tf.placeholder(shape=input_shape, dtype=tf.float32, name="D_in")

        with tf.variable_scope('G_OUT'):
            diff = Generate.age_G(data=G_image, age_label=G_label, is_training=is_training)
            G_fake = diff + G_image
            tf.summary.image('Gout_image', G_fake)

        with tf.variable_scope('D_OUT'):
            D_Tprob = Discriminate.age_D(data=D_image, age_label=D_label, is_training=is_training)

            D_Fprob = Discriminate.age_D(data=G_fake, age_label=D_label, is_training=is_training)


        with tf.variable_scope('Loss'):

            Euc_loss = tf.nn.l2_loss(G_image-G_fake)/(112*112*3)
             d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Tprob, labels=tf.ones_like(D_Tprob))
            # d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Fprob, labels=tf.zeros_like(D_Fprob))
            # D_loss = tf.reduce_mean(d_loss_real+d_loss_fake)
            # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_Fprob, labels=tf.ones_like(D_Fprob)))+Euc_loss

            D_loss = tf.reduce_sum(tf.square(D_Tprob - 1) + tf.square(D_Fprob)) / 2
            G_loss = tf.reduce_sum(tf.square(D_Fprob - 1)) / 2 + Euc_loss
        with tf.variable_scope('LR'):
            G_learning_rate = tf.train.exponential_decay(
                learning_rate=Lr_G,
                global_step=global_step,
                decay_steps=2000,
                decay_rate=0.5,
                staircase=True
            )
            D_learning_rate = tf.train.exponential_decay(
                learning_rate=Lr_D,
                global_step=global_step,
                decay_steps=2000,
                decay_rate=0.5,
                staircase=True
            )

        g_var = [var for var in tf.global_variables() if 'Generate' in var.name]
        d_var = [var for var in tf.global_variables() if 'Discriminator' in var.name]


        #D_train = tf.train.MomentumOptimizer(learning_rate=D_learning_rate,momentum=0.9).minimize(D_loss, var_list=d_var,global_step=global_step)
        D_train = tf.train.AdamOptimizer(learning_rate=D_learning_rate, beta1=0.5).minimize(D_loss, var_list=d_var, global_step=global_step)
        G_train = tf.train.AdamOptimizer(learning_rate=G_learning_rate, beta1=0.5).minimize(G_loss, var_list=g_var, global_step=global_step)

        # G_train = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=g_var,
        #                                                                                     global_step=global_step)
        # D_train = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=d_var,
        #                                                                                     global_step=global_step)

        saver = tf.train.Saver(var_list=tf.global_variables())

        data_generate = Data_process(mode='train', batch_size=batch, shuffle=True)
        Test_generate = Data_process(mode='train',  batch_size=batch, shuffle=True)

        #d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_var]
        with tf.Session() as sess:

            if not os.path.exists(Test_imgpath):
                os.makedirs(Test_imgpath)

            if not os.path.exists(event_log_path):
                os.makedirs(event_log_path)

            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            sess.run(tf.global_variables_initializer())

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            for s in range(Max_step):
                data = data_generate.get_next_batch()
                G_data = [x[0] for x in data]
                D_data = [x[1] for x in data]

                G_img = [x[0] for x in G_data]
                #G_group = [x[1] for x in G_data]

                D_img = [x[0] for x in D_data]
                D_group = [x[1] for x in D_data]

                gin_label = produce_label(Gage_shape, D_group, batch)
                dis_label = produce_label(Dage_shape, D_group, batch)


                Tdata = Test_generate.get_next_batch()

                TG_data = [x[0] for x in Tdata]
                TD_data = [x[1] for x in Tdata]

                TG_img = [x[0] for x in TG_data]
                TG_group = [x[1] for x in TG_data]

                TD_img = [x[0] for x in TD_data]
                TD_group = [x[1] for x in TD_data]


                Tgin_label = produce_label(Gage_shape, TD_group, batch)
                Tdis_label = produce_label(Dage_shape, TD_group, batch)

               # sess.run(d_clip)
                _, D_loss_value = sess.run([D_train, D_loss], feed_dict={D_image: D_img, D_label: dis_label,
                                                                                             G_image: G_img, G_label: gin_label, is_training: True})
                _, G_loss_value, diff_img = sess.run([G_train, G_loss, diff],
                                                                    feed_dict={D_image: D_img, D_label: dis_label, G_image: G_img,
                                                                               G_label: gin_label, is_training: True})

                print("Generator Loss: {}".format(G_loss_value))
                print("Discriminate Loss: {}\n".format(D_loss_value))

                if s % 100 == 0:
                    all_summary_obj = tf.summary.merge_all()
                    Event_writer = tf.summary.FileWriter(logdir=event_log_path, graph=sess.graph)
                    TG_loss,Fake_img, E_loss, all_summaries = sess.run([G_loss, G_fake, Euc_loss, all_summary_obj], feed_dict={D_image: TD_img, D_label: Tdis_label, G_image: TG_img,
                                                                                   G_label: Tgin_label, is_training: False})
                    print("Generator Loss: {}\n".format(TG_loss))
                    print("Euc_loss:{}".format(E_loss))

                    Save_batch_IMG(batch_img=Fake_img, org_img=TG_img,
                                   in_label=TG_group, out_label=TD_group, img_path=Test_imgpath, step=s)

                    Event_writer.add_summary(summary=all_summaries, global_step=s)
                    variables_save_path = os.path.join(checkpoint_path, 'model-parameters.bin')
                    saver.save(sess, variables_save_path, global_step=s)



if __name__ == '__main__':
    train()
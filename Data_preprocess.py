import numpy as np
from scipy import misc
import random
Base_path = r"/home/kris/PycharmProjects/FAG/"
image_path = Base_path + "file_index/"


class Image_dataGenerator():



    def __init__(self, batch=16, mode='train', classes=4):
        if mode == 'train':
            self.group_indexs = ['train_AgeBelow30.txt',
                         'train_AgeBelow40.txt',
                         'train_AgeBelow50.txt',
                         'train_AgeHigher50.txt',]
        else:
            self.group_indexs = ['test_AgeBelow30.txt',
                         'test_AgeBelow40.txt',
                         'test_AgeBelow50.txt',
                         'test_AgeHigher50.txt',]
        self.group_labels = ['B30', 'B40', 'B50', 'H50']
        self.data_size = [0, 0, 0, 0]
        self.classes = classes
        self.batch = batch
        self.data_list = []
        self.data_index = []
        self.get_data_list()
        self.counter = 0


    def process_image(self,image):
        img = (image - 127.5)/128
        return img

    def reset_counter(self):
        self.counter = 0

    def get_data_list(self):
        for index, labels in enumerate(self.group_indexs):
            with open(image_path+labels, encoding="ISO-8859-1")as f:
                data = np.loadtxt(f, dtype=str)
                self.data_list = np.concatenate((self.data_list,data[:,1]))
                self.data_size[index] = len(data)
        self.data_index = range(len(self.data_list))
        for i in reversed(range(len(self.data_size))):
            for j in range(0, i):
                self.data_size[i] = self.data_size[j]+self.data_size[i]


    def data_loader(self, file_name):
        data = []
        for i in range(self.batch):
            img_path = file_name[i]
            img = misc.imread(img_path)
            img = self.process_image(img)
            data.append(img)

        return data

    def get_classes(self, index):
        if index <= self.data_size[0]:
            current_index = 0
        elif self.data_size[0] < index & index <= self.data_size[1]:
            current_index = 1
        elif self.data_size[1] < index & index <= self.data_size[2]:
            current_index = 2
        else:
            current_index = 3

        return current_index


    def get_next_batch(self):

        dis_label = []
        gen_start = self.counter
        gen_end = self.counter+self.batch


        sindex = self.get_classes(gen_start)
        eindex = self.get_classes(gen_end)


        if sindex == eindex:
            gen_inlist = self.data_list[gen_start:gen_end]

        elif sindex == 0 & sindex != eindex:

            pad_index = random.sample(range(0, self.data_size[sindex]),
                          self.batch - self.data_size[sindex] + gen_start)


            pad_list = [self.data_list[x] for x in pad_index]

            gen_inlist = np.concatenate((self.data_list[gen_start:self.data_size[sindex]],
                                pad_list))
        else:
            pad_index = random.sample(range(self.data_size[sindex-1], self.data_size[sindex]),
                                      self.batch - self.data_size[sindex] + gen_start)


            pad_list = [self.data_list[x] for x in pad_index]

            gen_inlist = np.concatenate((self.data_list[gen_start:self.data_size[sindex]],
                                         pad_list))

        last_index = np.delete(self.data_index, range(gen_start, gen_end))
        dis_index = np.random.choice(last_index, self.batch,
                                     replace=False)


        dis_inlist = [self.data_list[x] for x in dis_index]


        for i in range(len(dis_index)):
            dis_label.append(self.get_classes(dis_index[i]))
        gen_label = np.zeros((self.batch))
        gen_label[:] = sindex

        gen_data = self.data_loader(file_name=gen_inlist)
        dis_data = self.data_loader(file_name=dis_inlist)

        self.counter = self.counter+self.batch
        if self.counter < len(self.data_list) & \
                self.counter + self.batch > len(self.data_list):
            self.counter = len(self.data_list) - self.batch
        elif self.counter>len(self.data_list):
            self.reset_counter()

        return [gen_data, gen_label], [dis_data, dis_label]




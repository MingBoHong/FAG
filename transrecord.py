import tensorflow as tf
import numpy as np
from scipy import misc
Base_path = r"/home/kris/FAG/"
image_path = Base_path + "cacd/"
labels = ['B30', 'B40', 'B50', 'H50']
import random


def file2group():
    inf_list = []
    Alldata = dict()
    for label in labels:
        Alldata[label] = []

    with open("young.lst")as f:
        for lists in f.readlines():
            line = lists.replace("\t", " ").replace('\n', '').split(" ")
            inf_list.append([line[1],
                             line[2].replace("./data/cacd/", ""), line[3]])

    for i in range(len(inf_list)):
        age = int(inf_list[i][0])
        sex = int(inf_list[i][2])
        if age <= 30:
            Alldata['B30'].append([age, image_path + inf_list[i][1], sex])
        elif 30 < age & age <= 40:
            Alldata['B40'].append([age, image_path + inf_list[i][1], sex])
        elif 40 < age & age <= 50:
            Alldata['B50'].append([age, image_path + inf_list[i][1], sex])
        else:
            Alldata['H50'].append([age, image_path + inf_list[i][1], sex])
    for label in labels:
        np.random.shuffle(Alldata[label])

    B30train = Alldata['B30'][:int(len(Alldata['B30'])*0.9)]
    B30Test = Alldata['B30'][int(len(Alldata['B30'])*0.9):]
    B40train = Alldata['B40'][:int(len(Alldata['B40']) * 0.9)]
    B40Test = Alldata['B40'][int(len(Alldata['B40']) * 0.9):]
    B50train = Alldata['B50'][:int(len(Alldata['B50']) * 0.9)]
    B50Test = Alldata['B50'][int(len(Alldata['B50']) * 0.9):]
    H50train = Alldata['H50'][:int(len(Alldata['H50']) * 0.9)]
    H50Test = Alldata['H50'][int(len(Alldata['H50']) * 0.9):]
    np.savetxt("file_index/train_AgeBelow30.txt", B30train, fmt="%s")
    np.savetxt("file_index/train_AgeBelow40.txt", B40train, fmt="%s")
    np.savetxt("file_index/train_AgeBelow50.txt", B50train, fmt="%s")
    np.savetxt("file_index/train_AgeHigher50.txt", H50train, fmt="%s")
    np.savetxt("file_index/test_AgeBelow30.txt", B30Test, fmt="%s")
    np.savetxt("file_index/test_AgeBelow40.txt", B40Test, fmt="%s")
    np.savetxt("file_index/test_AgeBelow50.txt", B50Test, fmt="%s")
    np.savetxt("file_index/test_AgeHigher50.txt", H50Test, fmt="%s")


class Image_dataGenerator():
    def __init__(self, batch, mode='train', classes=4):
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
        self.counter = [0, 0, 0, 0]
        self.data_size = [0, 0, 0, 0]
        self.get_datasize()
        self.classes = classes
        self.age_label_shape = [7,7,self.classes]
        self.batch = batch

    def process_image(self,image):
        img = (image - 127.5)/128
        return img

    def get_datasize(self):
        for index,list in enumerate(self.group_indexs):
            with open(list)as f:
                self.data_size[index] = len(f.readlines())

    def get_ImgPairsLists(self):
        # get random index from 4 group.Then, get one batch.

        index_lists = list(range(self.classes))
        young_list = random.randint(0, self.classes-1)

        index_lists.remove(young_list)
        old_list = random.sample(index_lists, 1)

        young_index = random.sample(range(0, self.data_size[young_list]), self.batch)
        old_index = random.sample(range(0, self.data_size[old_list]), self.batch)

        return young_list, young_index, old_list, old_index




    def data_loader(self, file_name, classes, index):
        pic = []
        label = np.zeros(self.age_label_shape)
        label[:][:][classes] = 1
        with open(file_name)as f:
            for lists in f.readlines():
                img = misc.imread(lists)
                img = self.process_image(img)
                pic.append([img, label])

        data = [pic[x] for x in index]

        return data


    def get_next_batch(self):
        y_list, y_index, o_list, o_index = self.get_ImgPairsLists()
        y_data  = self.data_loader(file_name=self.group_indexs[y_list],
                                   classes=y_list,index=y_index)
        o_data = self.data_loader(file_name=self.group_indexs[o_list],
                                  classes=o_list, index=o_index)


        return y_data,o_data





if __name__ == '__main__':
    f = open("file_index/train_AgeBelow30.txt",encoding="ISO-8859-1")
    data = np.loadtxt(f,dtype=str)

    print(data)





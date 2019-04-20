import numpy as np
import os
from scipy import misc

Base_path = r"/home/kris/PycharmProjects/FAG/IMG/"
image_path = r"/home/kris/FAG/cacd/"


class Data_process():

    def __init__(self, batch_size, fold=10, shuffle=False, split=False, mode='train'):

        self.IMG_list = []
        self.data_size = 0
        self.batch_size = batch_size
        self.counter = 0
        self.age_group = [0, 1, 2, 3]
        self.age_group_lst = dict()
        self.folder_name = ['00', '01', '10',
                            '11', '20', '21',
                            '30', '31']
        self.init_dict()
        self.fold = fold
        if os.path.exists("trainLst.lst")is False or os.path.exists("trainLst.lst") is False:
            self.split_dataset(self.fold)
        if split is True:
            self.split_dataset(self.fold)
        if mode =='train':
            self.get_data_list(file_name="trainLst.lst")
        else:
            self.get_data_list(file_name='testLst.lst')

        self.get_group_lst()
        if shuffle is True:
            np.random.shuffle(self.IMG_list)


    def split_dataset(self,n):
        IMG_list = []
        with open("IMG.lst")as f:
            Img_Lists = f.readlines()
            for img in Img_Lists:
                data = img.replace("\n", "")
                IMG_list.append(data)
            np.random.shuffle(Img_Lists)
            lst_size = len(IMG_list)
            test_size = int(lst_size / n)
            test_lst = IMG_list[0:test_size]
            train_lst = IMG_list[test_size:]
            print(len(train_lst))
            np.savetxt("trainLst.lst", train_lst, fmt="%s")
            np.savetxt("testLst.lst", test_lst, fmt="%s")

    def init_dict(self):
        for val in self.folder_name:
            self.age_group_lst[val]=[]


    def reset_counter(self):
        self.counter = 0

    def get_data_list(self, file_name):

        f = open(file_name, encoding="ISO-8859-1")
        Img_Lists = np.loadtxt(f, dtype=str)
        self.IMG_list = Img_Lists
        self.data_size = len(Img_Lists)

    def get_group_lst(self):
        for img in self.IMG_list:
            IMG_gender = img[-1]
            IMG_Group = img[0]
            IMG_name = img[2]
            self.age_group_lst[str(IMG_Group)+str(IMG_gender)].append(IMG_name)

    def process_image(self, image):
        img = (image - 127.5) / 128
        return img

    def data_loader(self, img_path):
        img = misc.imread(img_path)
        img = self.process_image(img)
        return img


    def get_next_batch(self):

        one_batch = []

        for i in range(self.batch_size):
            list = self.IMG_list[self.counter]
            IMG_gender = list[-1]
            IMG_Group = list[0]
            IMG_name = list[2]
            Dage_group = np.random.choice(np.delete(self.age_group, IMG_Group),
                                          replace=False)  # choose an group except what G's input image belongs to
            group_index = str(Dage_group) + str(IMG_gender)
            DIMG_index = np.random.randint(0, len(
                self.age_group_lst[group_index]))  # Generate one IMG's index from the group randomly
            DIMG_name = self.age_group_lst[group_index][DIMG_index]

            GIMG = self.data_loader(IMG_name)
            DIMG = self.data_loader(DIMG_name)

            one_batch.append([[GIMG, IMG_Group], [DIMG, Dage_group]])
            self.counter = self.counter + 1


        self.counter = self.counter + self.batch_size
        if self.counter < self.data_size & self.counter + self.batch_size > self.data_size:
            self.counter = self.data_size - self.batch_size
        elif self.counter >= self.data_size:
            self.reset_counter()
        return one_batch

if __name__ == '__main__':
    IMG = Data_process(batch_size=10)

    for i in range(500):
        data = IMG.get_next_batch()
        G_data = [x[0] for x in data]
        D_data = [x[1] for x in data]

        G_img = [x[0]for x in G_data]
        G_group = [x[1] for x in G_data]

        print(G_group)




import numpy as np
import random

data_list = [1,2,3,4,5,6,7,8,9,10]

data_index = [4,5]

#data = random.sample([x for x in data_list if x not in data_list],4)



deld=[data_list[x]for x in data_index]

data = np.delete(data_list,deld)

print(np.random.choice(data,5,replace=False))




25

10   20

20
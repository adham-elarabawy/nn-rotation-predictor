import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def cifar_reshape(model_arg):
    return np.reshape(model_arg, (10000, 3, 32, 32))

for batch_num in range(1,6):
    path = 'data/cifar-10-batches-py/data_batch_' + str(batch_num)
    batch = unpickle(path)
    batch_data = cifar_reshape(batch[b'data'])
    batch_labels = batch[b'labels']

    for i, raw_img in enumerate(batch_data):
        print(i, end='\r')
        # messing around with channels to output np
        img = np.rot90(raw_img.T, 3)
        label = batch_labels[i]
        for r in range(4):
            img_rotated = np.rot90(img, r)
            img_save = Image.fromarray(img_rotated)
            img_save.save('data/unpacked/data_batch_c' + str(label) + '_r' + str(r) + '_b' + str(batch_num) + '_' + str(i) + '.png')

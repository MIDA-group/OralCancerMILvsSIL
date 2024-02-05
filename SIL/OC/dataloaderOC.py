import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from PIL import Image



class OC_SIL():
    def __init__(self, list_IDs, labels, transform, path, batch_size, dim, n_channels,
                 n_classes, negative_class_name, positive_class_name, shuffle):
        """Initialization"""
        self.IDs = []
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs # name of images without focus level part
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.negative_class_name = negative_class_name
        self.positive_class_name = positive_class_name
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = path
        self.transform = transform

    def __len__(self):
        """Denotes the number of batches per epoch"""
        self.dataset = self.list_IDs
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # print ' index '+str(index)
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = torch.empty(self.batch_size, self.n_channels, self.dim[0], self.dim[1], dtype=torch.float32)
        y = torch.empty(self.batch_size, dtype=torch.int64)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # print(list_IDs_temp[i], ID)
            exists = os.path.isfile(os.path.join(self.path, self.negative_class_name, ID))
            if exists:
                image = Image.open(os.path.join(self.path, self.negative_class_name, ID))
                image_tensor = self.transform(image)
                # image = cv2.imread(os.path.join(self.path, self.negative_class_name, ID))
                # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # augmented = self.transform(img)#(image=img)
                # image_tensor = augmented['image']
                # focus_stack = torch.cat((focus_stack, image_tensor), 0)
            else:
                image = Image.open(os.path.join(self.path, self.positive_class_name, ID))
                image_tensor = self.transform(image)
                # image = cv2.imread(os.path.join(self.path, self.positive_class_name, ID))
                # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # augmented = self.transform(img)#(image=img)
                # image_tensor = augmented['image']
            X[i, ] = image_tensor
            y[i] = self.labels[ID]
        return X, y
    
    

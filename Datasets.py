#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:12:04 2020

@author: zhi
"""


import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import matplotlib.pyplot as plt



class ImageTSDataset_PHM(Dataset):
    
    def __init__(self, ImageDataFoloder, transform=None, condition="50Hz_High"):
        self.ImageDataFolder = ImageDataFoloder
        self.ImageDataList = os.listdir(ImageDataFoloder)
        
        os.chdir(ImageDataFoloder)
        self.ImageDataList = sorted(os.listdir(ImageDataFoloder))
        
        self.transform = transform
        
        self.labelDict = {"helical 1_" + condition + "_1": 0, "helical 1_" + condition + "_2": 0,
                          "helical 2_" + condition + "_1": 1, "helical 2_" + condition + "_2": 1,
                          "helical 3_" + condition + "_1": 2, "helical 3_" + condition + "_2": 2,
                          "helical 4_" + condition + "_1": 3, "helical 4_" + condition + "_2": 3,
                          "helical 5_" + condition + "_1": 4, "helical 5_" + condition + "_2": 4,
                          "helical 6_" + condition + "_1": 5, "helical 6_" + condition + "_2": 5,
                          "spur 1_" + condition + "_1": 6, "spur 1_" + condition + "_2": 6}
        
        self.numClasses = int(len(self.labelDict) / 2)
        
    def __getitem__(self, idx):
        
        dataName = self.ImageDataList[idx]
        img = np.load(dataName, allow_pickle=True, encoding='latin1')
        img = np.expand_dims(img, axis=0)
    
        dataIden = dataName.split('.')[0]
        label = dataIden.split('__')[0]      

        if label in self.labelDict.keys():
            label = self.labelDict[label]
        else:
            label = 100
        
        return img, label

    def __len__(self):

        return len(self.ImageDataList)


class ImageTSDataset_Paderborn(Dataset):

    def __init__(self, ImageDataFoloder, transform=None):
        self.ImageDataFolder = ImageDataFoloder
        self.ImageDataList = os.listdir(ImageDataFoloder)

        os.chdir(ImageDataFoloder)
        self.ImageDataList = sorted(os.listdir(ImageDataFoloder))

        self.transform = transform
        self.labelDict = {'KA01': 0, 'KA03': 1, 'KA04': 2, 'KA05': 3,
                          'KA06': 4, 'KA07': 5, 'KA08': 6, 'KA09': 7,
                          'KA15': 8, 'KA16': 9, 'KA22': 10, 'KA30': 11,
                          'KB23': 12}

        #        self.labelDict = {'K001': 0, 'K002': 1, 'K003': 2, 'K004': 3, 'K005': 4,
        #                          'K006': 5, 'KA01': 6, 'KA03': 7, 'KA04': 8, 'KA05': 9,
        #                          'KA06': 10, 'KA07': 11, 'KA08': 12, 'KA09': 13, 'KA15': 14,
        #                          'KA16': 15, 'KA22': 16, 'KA30': 17, 'KB23': 18, 'KB24': 19,
        #                          'KB27': 20,  'KI01': 21, 'KI03': 22, 'KI04': 23, 'KI05': 24,
        #                          'KI07': 25, 'KI08': 26, 'KI14': 27, 'KI16': 28, 'KI17': 29,
        #                          'KI18': 30, 'KI21': 31}

        self.numClasses = len(self.labelDict)

    def __getitem__(self, idx):

        data_name = self.ImageDataList[idx]
        # img = plt.imread(dataName)
        img = np.load(data_name, allow_pickle=True, encoding='latin1')
        img = np.expand_dims(img, axis=0)

        data_iden = data_name.split('.')[0]
        label = data_iden.split('_')[3]  # KXX
        if label in self.labelDict.keys():
            label = self.labelDict[label]
        else:
            label = 100

        return img, label

    def __len__(self):

        '''
        In default, the first part of the data folder splited using '_'
        is the class label, e.g., 'KA01'.

        The folder heirarchy is
        - Paderborn
          - dataImage
            - NXX_MXX_KXX
        '''

        return len(self.ImageDataList)


if __name__ == '__main__':
    
    ImageDataFoloder = "/home/users/j/jiawen/EVT/phm/class0_14_50Hz_High"
    dataset = ImageTSDataset_PHM(ImageDataFoloder)
    transform = transforms.Compose([transforms.ToTensor()])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    for idx, (img, label) in enumerate(test_loader):
        print(label)
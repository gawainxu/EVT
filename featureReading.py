#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:57:45 2020

@author: zhi

References:
https://zhuanlan.zhihu.com/p/75054200
https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
"""


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset

from CNNs import LeNet_enhanced2
from Datasets import ImageTSDataset_PHM, ImageTSDataset_Paderborn


def getArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_dim", type=int, default=64)
    parser.add_argument("--test_datapath", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--if_cuda", type=bool, default=True)
    opt = parser.parse_args()

    if "phm" in opt.test_datapath:
        opt.dataset = "phm"
    else:
        opt.dataset = "paderborn"

    return opt


class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    
    def remove(self):    
        self.hook.remove()
        

def readFeatures(model, layerName, inData, num_classes):
    
    hookF = []
    modules = model.named_children()
    
    for name, module in modules:
        if name in layerName:
            hookF.append(Hook(module))
    
    out = model(inData)
    out.backward(torch.ones((1, num_classes), dtype=torch.float), retain_graph=True)

    featureMaps = {}    
    
    for idx, h in enumerate(hookF):
        featureMaps[layerName[idx]] = h.output.detach().numpy()
    
    return featureMaps


def tSNEVisualize(inMat, nComponents):
    """
    The function used to visualize the high-dimensional hyper points 
    with t-SNE (t-distributed stochastic neighbor embedding)
    """
    inEmbedded = TSNE(n_components=nComponents).fit_transform(inMat)
    sns.scatterplot(inEmbedded[:,0], inEmbedded[:,1], legend='full')
    
    
# This class is used for testing !!!!!!!!!!!!!!
class toyNet(nn.Module):
    def __init__(self):
        super(toyNet, self).__init__()
        self.fc1 = nn.Linear(3, 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)
        self.relu2 = nn.ReLU()
        self.initialize()
        
    def initialize(self):
        self.fc1.weight = nn.Parameter(torch.Tensor([[1., 2., 3.],
                                                     [-4., -5., -6.],
                                                     [7., 8., 9.],
                                                     [-10., -11., -12.]]))
        self.fc1.bias = nn.Parameter(torch.Tensor([1.0, 2.0, 3.0, 4.0]))
        self.fc2.weight = nn.Parameter(torch.Tensor([[1., 2., 3., 4.]]))
        self.fc2.bias = nn.Parameter(torch.Tensor([1.0]))
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        return y
    
    
    
if __name__ == "__main__":
    

    opt = getArgs()
    
    # Prepare the dataset
    if "phm" in opt.dataset:
        dataset = ImageTSDataset_PHM(ImageDataFoloder=opt.data_folder, condition=opt.condition)
    else:
        dataset = ImageTSDataset_Paderborn(ImageDataFoloder=opt.data_folder)
    transform = transforms.Compose([transforms.ToTensor()])
    testDTLoader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    if torch.cuda.is_available() and opt.if_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    num_classes = dataset.numClasses
    model = LeNet_enhanced2(opt.data_dim, num_classes)
    model.load_state_dict(torch.load(opt.model_path))

    """
    module names in LeNet: [conv1, pool1, conv2, pool2, linear1, linear2, outlayer, dropout]
    for name, module in model.named_children():
        print name
    """
    model = model.to(device)
    
    '''
    layersToSee = ["conv1", "conv2", "conv3", "linear1", "linear2"]
    allFeatures = {"conv1": [], "conv2": [], "conv3":[], "linear1": [], "linear2": [], "label": []}
    
    for idx, (data, label) in enumerate(testDTLoader):
        data = data.to(device, dtype=torch.float)
        featureMaps = readFeatures(model, layersToSee, data)
        allFeatures["conv1"].append(featureMaps["conv1"])
        allFeatures["conv2"].append(featureMaps["conv2"])
        allFeatures["conv3"].append(featureMaps["conv3"])
        allFeatures["linear1"].append(featureMaps["linear1"])
        allFeatures["linear2"].append(featureMaps["linear2"])
        allFeatures["label"].append(label)
    '''

    layersToSee = ["output"]
    allFeatures = {"output": [], "label": []}
    
    for idx, (data, label) in enumerate(testDTLoader):
        print(idx, label)
        data = data.to(device, dtype=torch.float)
        featureMaps = readFeatures(model, layersToSee, data, num_classes)
        allFeatures["linear3"].append(featureMaps['linear3'])
        allFeatures["label"].append(label)

    with open(opt.save_path, "wb") as ff:
        pickle.dump(allFeatures, ff)
    
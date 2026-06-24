#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:15:08 2020

@author: zhi
"""


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import numpy as np
import pickle

from CNNs import LeNet_enhanced2
from Datasets import ImageTSDataset


def test(model, device, dataLoader):
    model.eval()
    model = model.to(device)
    outputs = []
    labels = []
    
    for idx, (img, label) in enumerate(dataLoader):
        #print idx
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        output = model(img)
        
        outputs.append(output.detach().numpy())
        labels.append(label.item())
        #lossTest.append(loss.cpu().detach().numpy())
        
    return outputs, labels


def compareLabels(estLabels, trueLabels):
    
    assert len(estLabels) == len(trueLabels)
    unEquals = 0
    for i in range(len(estLabels)):
        if estLabels[i] != trueLabels[i]:
            unEquals += 1
            
    return unEquals


def PrecisionRecall(outputs, labels):
            
    recalls = []
    precisions = []
    classes = range(14)
    classes.append(100)
    
    for c in classes:
        if c == 100:
            recalls.append(0)
            precisions.append(0)
            continue
        sampleIdx = np.where(labels == c)
        tp = np.sum(outputs[sampleIdx] == c)
        fp = np.sum(outputs == c) - tp
        fn = np.sum(labels == c) - tp
        recalls.append(tp*1.0 / (tp + fn))
        precisions.append(tp*1.0 / (tp+fp))
        
    precision = np.mean(np.array(precisions))
    recall = np.mean(np.array(recalls))
        
    return precision, recall


if __name__ == '__main__':
    
    N = 64
    numClasses = 14
    
    testDataFolder = '/home/zhi/projects/faultDiagnosis/phm/class0_28_50hz_Low_3200_end/'
    testDT = ImageTSDataset(testDataFolder)
    
    testDTLoader = DataLoader(testDT, batch_size=1 , shuffle=True, drop_last=True)
    device = 'cpu'
    
    modelPath = '/home/zhi/projects/faultDiagnosis/phm/LossFiles/LeNet_enhanced2_class0_14_50hz_Low.pt'
    model = LeNet_enhanced2(N, numClasses)
    model.load_state_dict(torch.load(modelPath))
    
    outputs, labels = test(model, device, testDTLoader)
    
    # read the outputs
    outputSorted = [[] for _ in range(numClasses + 1)]      # 1 for the outlier
    predictSorted = [[] for _ in range(numClasses + 1)]
    
    for (outs, label) in zip(outputs, labels):
        print label
        predict = F.log_softmax(torch.tensor(outs))
        predict = torch.argmax(predict, dim=1)
        print predict.item()
        if label < 100:   #predict == label:
            outputSorted[label].append(outs)
            predictSorted[label].append(predict.item())
        else:
            outputSorted[numClasses].append(outs)   #continue
            predictSorted[numClasses].append(predict.item())
        #outputSorted[predict.item()].append((outs, label))
        
        
    with open('/home/zhi/projects/EVT/FeatureMaps/class0_14_50hz_Low_outputs', "wb") as f1:
        pickle.dump(outputSorted, f1)
        
#    with open('/home/zhi/projects/EVT/FeatureMaps/class0_28_50hz_Low_3200_end_outputs', "wb") as f2:
#        pickle.dump(outputSorted, f2)
#     
#    with open('/home/zhi/projects/EVT/FeatureMaps/class0_28_50hz_Low_3200_end_predicts', "wb") as f2:
#        pickle.dump(predictSorted, f2)
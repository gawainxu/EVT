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
import argparse
from sklearn.metrics import roc_curve, auc

from CNNs import LeNet_enhanced2
from Datasets import ImageTSDataset_PHM, ImageTSDataset_Paderborn


def getParse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument("--if_cuda", type=bool, default=True)
    parser.add_argument('--condition', type=str, default="50hz_High")
    parser.add_argument('--save_path', type=str, default=None)

    opt = parser.parse_args()

    if "phm" in opt.data_folder:
        opt.dataset = "phm"
    else:
        opt.dataset = "paderborn"

    return opt


def test(model, device, dataLoader):
    model.eval()
    model = model.to(device)
    outputs = []
    labels = []
    preds = []
    correct = 0
    
    for idx, (img, label) in enumerate(dataLoader):
        #print idx
        img = img.to(device, dtype=torch.float)
        output = model(img)
        pred = torch.argmax(output, dim=1)
        
        outputs.append(output.cpu().detach().numpy())
        pred = pred.cpu().detach().numpy()
        label = label.cpu().item()
        preds.append(pred)
        labels.append(label)
        if pred == label:
            correct += 1
    print("Acc is", correct / len(dataLoader))
        
    return outputs, preds, labels


def compareLabels(estLabels, trueLabels):
    
    assert len(estLabels) == len(trueLabels)
    unEquals = 0
    for i in range(len(estLabels)):
        if estLabels[i] != trueLabels[i]:
            unEquals += 1
            
    return unEquals


def PrecisionRecall(preds, labels, num_classes):
            
    recalls = []
    precisions = []
    classes = list(range(num_classes))
    classes.append(100)

    preds = np.array(preds)
    #print(len(preds))
    #print(len(labels))
    print(preds.shape)
    
    for c in classes:
        if c == 100:
            recalls.append(0)
            precisions.append(0)
            continue
        class_c_idx = [i for i, x in enumerate(labels) if x == c]
        print(len(class_c_idx))
        tp = len(np.where(preds[class_c_idx] == c)[0])
        fp = len(np.where(preds == c)[0]) - tp
        fn = len(class_c_idx) - tp
        print("tp", tp, "fp", fp, "fn", fn)
        recalls.append(tp*1.0 / (tp + fn))
        precisions.append(tp*1.0 / (tp+fp))
        
    precision = np.mean(np.array(precisions))
    recall = np.mean(np.array(recalls))

    correct = 0
    for i, j in zip(preds, labels):
        if i == j:
            correct += 1
    accuracy = correct / len(labels)
        
    return precision, recall, accuracy


def AUROC(labels, probs):
    '''
    ROC:
        X: False positive rate
        Y: True positive rate
    '''
    fpr, tpr, threholds = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)

    return auroc


if __name__ == '__main__':


    opt = getParse()

    #os.chdir(opt.dataFolder)
    if "phm" in opt.dataset:
        dataset = ImageTSDataset_PHM(ImageDataFoloder=opt.data_folder, condition=opt.condition)
    else:
        dataset = ImageTSDataset_Paderborn(ImageDataFoloder=opt.data_folder)
    transform = transforms.Compose([transforms.ToTensor()])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    num_classes = dataset.numClasses

    if torch.cuda.is_available() and opt.if_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = LeNet_enhanced2(opt.in_dim, num_classes)
    model.load_state_dict(torch.load(opt.model_path))
    outputs, preds, labels = test(model, device, test_loader)
    precision, recall, accuracy = PrecisionRecall(preds, labels, num_classes)
    print("Precision:", precision, "Recall:", recall, "accuracy", accuracy)
    
    # read the outputs
    output_sorted = [[] for _ in range(num_classes + 1)]      # 1 for the outlier
    predict_sorted = [[] for _ in range(num_classes + 1)]
    
    for (outs, label) in zip(outputs, labels):
        #print(label)
        predict = F.log_softmax(torch.tensor(outs))
        predict = torch.argmax(predict, dim=1)
        #print(predict.item())
        if label < 100:   #predict == label:
            output_sorted[label].append(outs)
            predict_sorted[label].append(predict.item())
        else:
            output_sorted[num_classes].append(outs)   #continue
            predict_sorted[num_classes].append(predict.item())
        #outputSorted[predict.item()].append((outs, label))
        
        
    with open(opt.save_path, "wb") as f1:
        pickle.dump(output_sorted, f1)
        
#    with open('/home/zhi/projects/EVT/FeatureMaps/class0_28_50hz_Low_3200_end_outputs', "wb") as f2:
#        pickle.dump(outputSorted, f2)
#     
#    with open('/home/zhi/projects/EVT/FeatureMaps/class0_28_50hz_Low_3200_end_predicts', "wb") as f2:
#        pickle.dump(predictSorted, f2)
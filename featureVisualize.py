#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:57:45 2020

@author: zhi
"""


import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle 

import torch
import torch.nn as nn



def pca(inMat, nComponents):
    
    # It is better to make PCA transformation before tSNE
    pcaFunction = PCA(nComponents)
    outMat = pcaFunction.fit_transform(inMat)

    return outMat    
    
    

def tSNE(inMat, nComponents):
    """
    The function used to visualize the high-dimensional hyper points 
    with t-SNE (t-distributed stochastic neighbor embedding)
    https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0
    https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    """
    
    inEmbedded = TSNE(n_components=nComponents, perplexity=10).fit_transform(inMat)
    return inEmbedded
    
    
    
if __name__ == "__main__":
    
    with open('/home/zhi/projects/EVT/FeatureMaps/class0_28_30hz_High_3200_end_outputs', "rb") as ff:
       allFeatures = pickle.load(ff)
    
    outputs = []
    labels = []
    allOutputs = []
    for i, f in enumerate(allFeatures):
        allOutputs = allOutputs + f
        labels += len(f)*[i]
    
    allOutputs = np.array(allOutputs)
    labels = np.array(labels)
    
    allOutputs = allOutputs.squeeze()
    
    numVisu = 10000
    #nCompPCA = 6
    #outputsPCA = pca(outputs, nCompPCA)
    
    visuIdx = np.random.choice(len(labels), numVisu)
    allOutputs = allOutputs[visuIdx, :]
    labels = labels[visuIdx]
    
    outputsTSNE = tSNE(allOutputs, nComponents=2)
    
    outputsTSNE0 = outputsTSNE[:, 0]
    outputsTSNE1 = outputsTSNE[:, 1]
    
    outputsTSNE0 = outputsTSNE0.reshape(outputsTSNE0.shape[0])
    outputsTSNE1 = outputsTSNE1.reshape(outputsTSNE1.shape[0])
    labels = labels.reshape(labels.shape[0])
    
    
    # start the visualization
    f = {"outputsTSNE_1": outputsTSNE0, "outputsTSNE_2": outputsTSNE1,
         "label": labels}
    fp = pd.DataFrame(f)
    
    sns.scatterplot(x="outputsTSNE_1", y="outputsTSNE_2", hue="label",
                    palette=['green','orange','brown','dodgerblue','red', 'yellow',  'pink', 'purple', 
                             'blue', 'c', 'grey', 'chocolate', 'deepskyblue', 'olive', 'black'], data=fp,
                    legend="full", alpha=0.5)
    

'''
'green','orange','brown','dodgerblue','red', 'yellow', 'pink', 'purple', 
                             'blue', 'c', 'grey', 'chocolate', 'orange', 'olive', 'olivedrab', 'darkseagreen',
                             'lime', 'turquoise', 'darkslategray', 'powderblue', 'deepskyblue', 'steelblue', 'stategray', 'royalblue',
                             'midnightblue', 'slateblue', 'mediumpurple', 'indigo', 'thistle', 'deeopink', 'crimson'
'''
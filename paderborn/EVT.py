#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:23:17 2021

@author: zhi
"""

import pickle
import libmr
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

numClasses = 13

def getMeans(outputs):

    nClasses = len(outputs)
    meanS = []
    for c in range(nClasses):
        outputC = np.squeeze(np.array(outputs[c]))
        meanC = np.mean(outputC, axis=0)
        meanS.append(meanC)
    
    return meanS


def getResiduals(meanS, outputs):
    
    nClasses = len(outputs)
    residuals = []
    for c in range(nClasses):
        residual = outputs[c] - meanS[c]
        residuals.append(residual)
        
    return residuals
        

def calcDistance(meanS, outputS, metric):
    
    distanceS = []
    for c in range(len(meanS)):
        outputs = outputS[c]
        mean = meanS[c]
        if metric == "euclidean":
            distances = [distance.euclidean(mean, d) for d in outputs]
            distanceS.append(distances)
        
    return distanceS
    
    
def fitEVT(distanceS, meanS, tailS):
    
    numClasses = len(meanS)
    mrS = []
    xs = np.linspace(0, 200, 100)
    fig, (ax1, ax2) = plt.subplots(2,1)
    for c in range(numClasses):
        mr = libmr.MR()
        tail = tailS[c]
        distances = distanceS[c]
        ax1.hist(distances,bins=20)
        mr.fit_high(distances, tail)
        assert mr.is_valid
        mrS.append(mr)
        ax2.plot(xs, mr.w_score_vector(xs), label="Tailsize")
        
    return mrS


def testEVT(mr, d):
    
    d = np.array([d])
    d = d.astype(np.double)
    score = mr.w_score_vector(d)
        
    return score
            

def plotDistance(distances):
    
    plt.hist(distances, bins=20)
    

def calcAccuracy(scoreS, labelS, T):
       
    # assume that the inliers are positive
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    
    for s, l in zip(scoreS, labelS):
        if s > T and l == numClasses:
            trueNegative += 1
        elif s > T and l != numClasses:
            falseNegative += 1
        elif s <= T and l != numClasses:
            truePositive += 1
        elif s <= T and l == numClasses:
            falsePositive += 1
    
    return truePositive, trueNegative, falsePositive, falseNegative
        
    
if __name__ == "__main__":
    with open('/home/zhi/projects/EVT/paderborn/FeatureMaps/class13_0_12_4_outputs', "rb") as ff:
        outputS = pickle.load(ff)
        
    meanS = getMeans(outputS)
    distanceS = calcDistance(meanS, outputS, "euclidean")
    
    tailS = [30 for i in range(len(meanS))]                   # tail size
    mrS = fitEVT(distanceS, meanS, tailS)
    
    with open('/home/zhi/projects/EVT/paderborn/FeatureMaps/classAll_16_18_4_outputs', "rb") as ff:
        outputSorted = pickle.load(ff)
        
    with open('/home/zhi/projects/EVT/paderborn/FeatureMaps/classAll_16_18_4_predicts', "rb") as ff:
        predictSorted = pickle.load(ff)
        
    labelS = []
    scoreS = []
    for c in range(numClasses + 1):
        outs = outputSorted[c]
        labels = len(outs)*[c]
        predicts = predictSorted[c]
        
        # inference the model here
        for i, sample in enumerate(outs):
            p = predicts[i]
            mean = meanS[p]
            mr = mrS[p]
           
            d = distance.euclidean(mean, sample)
            score = testEVT(mr, d)
        
            labelS.append(labels[i])
            scoreS.append(score)
        
    R = []
    P = []
    FPR = []
    for T in np.arange(0, 1, 0.05):
        truePositive, trueNegative, falsePositive, falseNegative = calcAccuracy(scoreS, labelS, T)
        precision = truePositive*1.0  / (truePositive + falsePositive)
        recall = truePositive*1.0 / (truePositive + falseNegative)
        FPR.append(falsePositive*1.0 / (falsePositive+trueNegative))
        R.append(recall)
        P.append(precision)
        
    plt.plot(FPR, R)
    
    F = []
    for i in range(20):
        F.append(2. * P[i] * R[i] / (P[i] + R[i]))
    
    smallIdx = np.argmin(np.array(F))
    print P[smallIdx], R[smallIdx], F[smallIdx]
    
    
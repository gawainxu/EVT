#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:50:44 2020

@author: zhi
"""

import os                
import numpy as np

           
N = 64
intervel = 64

labelDict = ["helical 1_45Hz_High_1", "helical 1_45Hz_High_2",
             "helical 2_45Hz_High_1", "helical 2_45Hz_High_2",
             "helical 3_45Hz_High_1", "helical 3_45Hz_High_2",
             "helical 4_45Hz_High_1", "helical 4_45Hz_High_2",
             "helical 5_45Hz_High_1", "helical 5_45Hz_High_2",
             "helical 6_45Hz_High_1", "helical 6_45Hz_High_2",
             
             "spur 1_45Hz_High_1", "spur 1_45Hz_High_2",
             "spur 2_45Hz_High_1", "spur 2_45Hz_High_2",
             "spur 3_45Hz_High_1", "spur 3_45Hz_High_2",
             "spur 4_45Hz_High_1", "spur 4_45Hz_High_2",
             "spur 5_45Hz_High_1", "spur 5_45Hz_High_2",
             "spur 6_45Hz_High_1", "spur 6_45Hz_High_2",
             "spur 7_45Hz_High_1", "spur 7_45Hz_High_2",
             "spur 8_45Hz_High_1", "spur 8_45Hz_High_2"]

selectList = range(0, 14)


dataFolder = './PHM_Society_2009_Competition_Expanded_txt/45Hz_High'
dataFolderList = sorted(os.listdir(dataFolder))
os.chdir(dataFolder)
distFolder = '/home/users/j/jiawen/EVT/phm/class0_28_45Hz_High'
if not os.path.exists(distFolder):
    os.mkdir(distFolder)


for i, fname in enumerate(dataFolderList):
    if i in selectList:
        continue
    print(fname)
    txtFile = fname + '/' + fname + '.txt'
    ts = []
    f = open(txtFile, 'r')
    lines = f.readlines()
    for l in lines:
        ws = l.split(' ')
        wsn = [w for w in ws if w != '']
        ts.append(float(wsn[1]))                       # TODO only the second channel
    dlen = len(ts)
            
    for idx, pointer in enumerate(range(0, dlen, intervel)):
        if pointer + N*N > dlen:
            break
        timeSeries = ts[pointer:pointer+N*N]
        timeSeries = np.array(timeSeries)
        timeSeries = timeSeries.reshape([N, N])
        timeSeries.dump(os.path.join(distFolder, fname + '__' + str(idx)+'.dat'))
            

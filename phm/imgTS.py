#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:50:44 2020

@author: zhi
"""

import argparse
import os                
import numpy as np


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--condition', type=str, required=True, default="50Hz_High", help="condition condition")

    return parser.parse_args()


opt = arg_parse()

N = 64
intervel = 64

labelDict = ["helical 1_" + opt.condition + "_1", "helical 1_" + opt.condition + "_2",
             "helical 2_" + opt.condition + "_1", "helical 2_" + opt.condition + "_2",
             "helical 3_" + opt.condition + "_1", "helical 3_" + opt.condition + "_2",
             "helical 4_" + opt.condition + "_1", "helical 4_" + opt.condition + "_2",
             "helical 5_" + opt.condition + "_1", "helical 5_" + opt.condition + "_2",
             "helical 6_" + opt.condition + "_1", "helical 6_" + opt.condition + "_2",
             
             "spur 1_" + opt.condition + "_1", "spur 1_" + opt.condition + "_2",
             "spur 2_" + opt.condition + "_1", "spur 2_" + opt.condition + "_2",
             "spur 3_" + opt.condition + "_1", "spur 3_" + opt.condition + "_2",
             "spur 4_" + opt.condition + "_1", "spur 4_" + opt.condition + "_2",
             "spur 5_" + opt.condition + "_1", "spur 5_" + opt.condition + "_2",
             "spur 6_" + opt.condition + "_1", "spur 6_" + opt.condition + "_2",
             "spur 7_" + opt.condition + "_1", "spur 7_" + opt.condition + "_2",
             "spur 8_" + opt.condition + "_1", "spur 8_" + opt.condition + "_2"]

selectList = range(0, 14)


dataFolder = './PHM_Society_2009_Competition_Expanded_txt/' + opt.condition
dataFolderList = sorted(os.listdir(dataFolder))
os.chdir(dataFolder)
distFolder = '/home/users/j/jiawen/EVT/phm/class0_28_' + opt.condition
if not os.path.exists(distFolder):
    os.mkdir(distFolder)


for i, fname in enumerate(dataFolderList):
    #if i in selectList:
    #    continue
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
            

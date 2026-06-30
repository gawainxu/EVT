import os
import argparse
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from CNNs import LeNet_enhanced2, train
from Datasets import ImageTSDataset


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFolder', type=str, default='/home/zhi/projects/datasets/fault diagnosis/phm/class0_14_50hz_High')
    parser.add_argument('--modelPath', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    
    args = parser.parse_args()
    
    return args



if __name__ == '__main__':
    
    opt = getArgs()
    
    os.chdir(opt.dataFolder)
    dataset = ImageTSDataset(opt.dataFolder)
    transform = transforms.Compose([transforms.ToTensor()])      
    imageDTLoader = DataLoader(dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 4, drop_last=True)
    
    numClasses = dataset.numClasses
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet_enhanced2(opt.in_dim, numClasses)
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    if opt.modelPath is not None:
        model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    
    scoreS = []
    labelS = []
    lossEpoch = 0
    
    for e in range(opt.epochs):
        loss_epoch = train(model, imageDTLoader, optimizer, device)
        print("Epoch", e, loss_epoch)

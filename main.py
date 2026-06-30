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
<<<<<<< HEAD
    parser.add_argument('--dataFolder', type=str, default='/home/zhi/projects/datasets/fault diagnosis/phm/class0_14_50hz_High')
=======
    parser.add_argument('--dataFolder', type=str, default='D:\projects\EVT\class0_14_30hz_High')
>>>>>>> 562c6c9151604a9501ffdea09079de203975c2eb
    parser.add_argument('--modelPath', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--if_cuda", type=bool, default=False)
    
    args = parser.parse_args()
    
    return args



if __name__ == '__main__':
    
    opt = getArgs()
    
    os.chdir(opt.dataFolder)
    dataset = ImageTSDataset(opt.dataFolder)
    transform = transforms.Compose([transforms.ToTensor()])      
    imageDTLoader = DataLoader(dataset, batch_size = opt.batch_size, shuffle=True, num_workers = 4, drop_last=True)
    
    numClasses = dataset.numClasses
<<<<<<< HEAD
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
=======

    if torch.cuda.is_available() and opt.if_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
>>>>>>> 562c6c9151604a9501ffdea09079de203975c2eb
    model = LeNet_enhanced2(opt.in_dim, numClasses)
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    if opt.modelPath is not None:
        model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    optimizer = optim.Adadelta(model.parameters(), lr = opt.lr)
    
    scoreS = []
    labelS = []
    lossEpoch = 0
    
    lossMin = 20
    for e in range(opt.epochs):
<<<<<<< HEAD
        loss_epoch = train(model, imageDTLoader, optimizer, device)
        print("Epoch", e, loss_epoch)
=======
        lossEpoch = train(model, device, imageDTLoader, optimizer)
      #  Loss.append(lossEpoch)
        print('Epoch: ', e, 'Loss: ', lossEpoch)
        if lossEpoch < lossMin:
            torch.save(model.state_dict(), '/home/zhi/projects/faultDiagnosis/phm/LossFiles/LeNet_enhanced2_class0_14_30hz_high.pt')
            lossMin = lossEpoch
>>>>>>> 562c6c9151604a9501ffdea09079de203975c2eb

import os
import argparse
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from CNNs import LeNet_enhanced2, train
from Datasets import ImageTSDataset


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFolder', type=str, default='/home/zhi/projects/faultDiagnosis/phm/class0_28_50hz_High')
    parser.add_argument('--modelPath', type=str, default='/home/zhi/projects/faultDiagnosis/phm/LeNet_enhanced2_50hz_High.pth')
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
    optimizer = optim.Adadelta(model.parameters(), lr = lr)
    model = LeNet_enhanced2(opt.in_dim, numClasses)
    if opt.modelPath is not None:
        model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    
    scoreS = []
    labelS = []
    
    for e in range(opt.epochs):
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            output = model(img.float())
            score, pred = torch.max(output.data, 1)
            
            scoreS.append(score.item()) 
            labelS.append(label)
import os
import argparse

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from CNNs import LeNet_enhanced2, train
from Datasets import ImageTSDataset_PHM, ImageTSDataset_Paderborn


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--oldmodel_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument("--if_cuda", type=bool, default=True)
    parser.add_argument('--condition', type=str, default="50Hz_High")
    
    opt = parser.parse_args()

    if "phm" in opt.data_folder:
        opt.dataset = "phm"
    else:
        opt.dataset = "paderborn"
    
    return opt



if __name__ == '__main__':
    
    opt = getArgs()
    
    os.chdir(opt.dataFolder)
    if "phm" in opt.dataset:
        dataset = ImageTSDataset_PHM(ImageDataFoloder=opt.data_folder, condition=opt.condition)
    else:
        dataset = ImageTSDataset_Paderborn(ImageDataFoloder=opt.data_folder)
    transform = transforms.Compose([transforms.ToTensor()])      
    imageDTLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    numClasses = dataset.numClasses

    if torch.cuda.is_available() and opt.if_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = LeNet_enhanced2(opt.in_dim, numClasses)
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
    if opt.oldmodel_path is not None:
        model.load_state_dict(torch.load(opt.oldmodel_path))
    model.eval()
    
    scoreS = []
    labelS = []
    lossEpoch = 0
    
    lossMin = 20
    for e in range(opt.epochs):
        lossEpoch = train(model, device, imageDTLoader, optimizer)
        print('Epoch: ', e, 'Loss: ', lossEpoch)
        if lossEpoch < lossMin:
            torch.save(model.state_dict(), opt.model_path)
            lossMin = lossEpoch
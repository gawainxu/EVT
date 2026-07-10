from CNNs import LeNet_enhanced2
from Datasets import ImageTSDataset

import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

def getParse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--old_model_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--in_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument("--if_cuda", type=bool, default=True)
    parser.add_argument('--condition', type=str, default="50Hz_High")

    args = parser.parse_args()

    return args


def train(model, device, dataLoader, optimizer):
    # Hint: (for cross_entropy) https://jbencook.com/cross-entropy-loss-in-pytorch/
    model.train()
    model = model.cuda()
    lossEpoch = 0

    for batchIdx, (img, label) in enumerate(dataLoader):
        # img, label = torch.from_numpy(img), torch.from_numpy(label)
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(img)
        # output = F.log_softmax(output)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        lossEpoch += loss.cpu()

    return lossEpoch / batchIdx


if __name__ == '__main__':

    opt = getParse()

    imageDT = ImageTSDataset(opt.data_folder)

    transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize(dataMean, dataStd)
    imageDTLoader = DataLoader(imageDT, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Prepare the network
    if opt.if_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    model = LeNet_enhanced2(opt.in_dim, opt.num_classes)
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)

    lossMin = 20
    for e in range(opt.epochs):
        lossEpoch = train(model, device, imageDTLoader, optimizer)
        #  Loss.append(lossEpoch)
        print('Epoch: ', e, 'Loss: ', lossEpoch)
        if lossEpoch < lossMin:
            torch.save(model.state_dict(), '/home/users/j/jiawen/EVT/save/paderborn_3.pth')
            lossMin = lossEpoch
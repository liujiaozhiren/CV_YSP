import csv

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import Nets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_img(root, filedir, transform=None, value_cnt=1):
    # Data loading
    with open(filedir, 'r') as f:
        lines = f.readlines()
    output = []
    for line in lines:
        linesplit = line.split('\n')[0].split('\t')
        addr = linesplit[0]
        target = torch.tensor(float(linesplit[value_cnt]))
        if not os.path.exists(os.path.join(root, addr)):
            continue
        img = Image.open(os.path.join(root, addr)).convert('RGB')

        if transform is not None:
            img = transform(img)

        output.append([img, target, addr])

    return output


def load_model(pretrained_dict, new):
    model_dict = new.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    new.load_state_dict(model_dict)


def valid(data, net):
    net.eval()
    with torch.no_grad():
        label = []
        pred = []
        out = []
        with tqdm(enumerate(data), total=len(data), postfix='valid') as tq:
            for i, (img, target, addr) in tq:
                if device == 'cuda':
                    img = img.cuda(non_blocking=True)
                    # target = target.cuda(non_blocking=True)
                output = net(img).squeeze(1)
                label.extend(target.cpu().numpy())
                pred.extend(output.cpu().numpy())
                out.append([addr[0], target.item(), output.item()])
                # print(i)
        # measurements
        label = np.array(label)
        pred = np.array(pred)
        mse = np.mean(np.square(label - pred))
        correlation = np.corrcoef(label, pred)[0][1]
        mae = np.mean(np.abs(label - pred))
        rmse = np.sqrt(np.mean(np.square(label - pred)))
        y_true_mean = np.mean(label)
        rae = np.sum(np.abs(label - pred)) / np.sum(np.abs(label - y_true_mean))
    net.train()
    return correlation, mae, rmse, mse, rae, out


def loss(output, target):
    return nn.MSELoss()(output, target)


def main(value_cnt=1):
    # net definition 
    net = Nets.AlexNet().to(device)
    # net = Nets.ResNet(block = Nets.BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()

    # load pretrained model
    load_model(torch.load('./models/alexnet.pth', map_location=torch.device(device), encoding='latin1'), net)
    # load_model(torch.load('./models/resnet18.pth'), net)

    # evaluate
    # net.eval()

    # loading data...
    root = '../data/0216/Images'
    valdir = '../data/0216/score.csv'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    dataset = read_img(root, valdir, transform=transform, value_cnt=value_cnt)
    train_data = dataset[:int(len(dataset) * 0.8)]
    valid_data = dataset[int(len(dataset) * 0.8):]
    train_data = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    valid_data = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=0)

    opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss = nn.MSELoss()
    best_correlation, best_mae, best_rmse, _, _, _ = valid(valid_data, net)
    early_stop = 0
    print('Initial:Correlation:{correlation:.4f}\t'
          'Mae:{mae:.4f}\t'
          'Rmse:{rmse:.4f}\t'.format(correlation=best_correlation, mae=best_mae, rmse=best_rmse))
    for epoch in range(5000):
        tmp_loss = 0
        with tqdm(enumerate(train_data), total=len(train_data), postfix='train') as tq:
            for i, (img, target, addr) in tq:
                if device == 'cuda':
                    img = img.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                output = net(img).squeeze(1)
                l = loss(output, target)
                opt.zero_grad()
                l.backward()
                opt.step()
                tmp_loss += l.item()
                tq.set_postfix(loss=tmp_loss / (i + 1))
        correlation, mae, rmse, _, _, _ = valid(valid_data, net)
        cnt = 0
        if mae < best_mae:
            best_mae = mae
            cnt += 1
        if rmse < best_rmse:
            best_rmse = rmse
            cnt += 1
        if correlation > best_correlation:
            best_correlation = correlation
            cnt += 1
        if cnt >= 2:  # save the best model
            early_stop = 0
            torch.save(net.state_dict(), f'best_model{value_cnt}.pth')
            print('Better Save :Loss:{mseloss:.4f}\tCorrelation:{correlation:.4f}\t'
                  'Mae:{mae:.4f}\t'
                  'Rmse:{rmse:.4f}\t'.format(mseloss=tmp_loss / len(train_data),
                                             correlation=correlation, mae=mae, rmse=rmse))
        else:
            early_stop += 1
            print('Worse Drop:Loss:{mseloss:.4f}\tCorrelation:{correlation:.4f}\t'
                  'Mae:{mae:.4f}\t'
                  'Rmse:{rmse:.4f}\t'.format(mseloss=tmp_loss / len(train_data),
                                             correlation=correlation, mae=mae, rmse=rmse))
        if early_stop >= 20:
            break

    ckpoints = torch.load(f'best_model{value_cnt}.pth', map_location=torch.device(device))
    net.load_state_dict(ckpoints)
    _data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    correlation, mae, rmse, mse, rae, out = valid(_data, net)

    col_name = ["", "aggressive", "smart", "trustworthy", "beauty"]
    filename = f"{col_name[value_cnt]}_result_mse({mse})_mae({mae})_rmse({rmse})_rae({rae}).csv"

    # 使用'with'语句打开文件，确保文件会被正确关闭
    with open(filename, 'w', newline='') as csvfile:
        # 创建一个csv.writer对象
        csvwriter = csv.writer(csvfile)
        # 写入数据
        for row in out:
            csvwriter.writerow(row)
    print('Correlation:{correlation:.4f}\t'
          'Mae:{mae:.4f}\t'
          'Rmse:{rmse:.4f}\t'.format(
        correlation=best_correlation, mae=best_correlation, rmse=best_correlation))




def atest(value_cnt=1):
    # net definition
    net = Nets.AlexNet().to(device)
    ckpoints = torch.load(f'best_model{value_cnt}.pth', map_location=torch.device(device))
    net.load_state_dict(ckpoints)
    net.eval()
    # loading data...
    root = '../data/0216/Images'
    valdir = '../data/0216/score.csv'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    dataset = read_img(root, valdir, transform=transform, value_cnt=value_cnt)
    _data = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    correlation, mae, rmse, mse, rae, out = valid(_data, net)
    col_name = ["", "aggressive", "smart", "trustworthy", "beauty"]
    filename = f"{col_name[value_cnt]}_result_mse({mse})_mae({mae})_rmse({rmse})_rae({rae}).csv"

    # 使用'with'语句打开文件，确保文件会被正确关闭
    with open(filename, 'w', newline='') as csvfile:
        # 创建一个csv.writer对象
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["addr", "target", "output"])
        # 写入数据
        for row in out:
            csvwriter.writerow(row)


if __name__ == '__main__':
    # btest(4)
    # main(2)
    atest(2)

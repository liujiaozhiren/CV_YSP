import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import Nets
import unittest
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image


class MyTestCase(unittest.TestCase):

    def valid(self, data, net, data_dict, value_cnt=4):
        net.eval()
        with torch.no_grad():
            label = []
            pred = []
            # out = []
            with tqdm(enumerate(data), total=len(data), postfix='valid') as tq:
                for i, (img, addr) in tq:
                    if device == 'cuda':
                        img = img.cuda(non_blocking=True)
                    output = net(img).squeeze(1)
                    pred.extend(output.cpu().numpy())
                    data_dict[addr[0]][2 + value_cnt] = output.item()
                    # print(i)
            # measurements
            #label = np.array(label)
            #pred = np.array(pred)
            #mse = np.mean(np.square(label - pred))
            # correlation = np.corrcoef(label, pred)[0][1]
            #mae = np.mean(np.abs(label - pred))
            #rmse = np.sqrt(np.mean(np.square(label - pred)))
            #y_true_mean = np.mean(label)
            #rae = np.sum(np.abs(label - pred)) / np.sum(np.abs(label - y_true_mean))
        net.train()
        # return mae, rmse, mse, rae

    def test(self):
        data_dict = {}
        data_dict2 = {}
        data_list = []
        root_dir = './faces'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    full_path = os.path.join(subdir, file)
                    parts = subdir.split('/')

                    img = Image.open(full_path).convert('RGB')
                    if transform is not None:
                        img = transform(img)
                    data_list.append([img, full_path])
                    data_dict[full_path] = [parts[-2], parts[-1], file, 0, 0, 0, 0]
                    if parts[-2]+'/'+parts[-1] not in data_dict2:
                        data_dict2[parts[-2]+'/'+parts[-1]] = {full_path:[0,0,0,0]}
                    else:
                        data_dict2[parts[-2]+'/'+parts[-1]][full_path] = [0,0,0,0]
        net = Nets.AlexNet().to(device)
        data = DataLoader(data_list, batch_size=1, shuffle=False, num_workers=0)
        col_name = ["", "aggressive", "smart", "trustworthy", "beauty"]
        for value_cnt in [1, 2, 3, 4]:
            ckpoints = torch.load(f'best_model{value_cnt}.pth', map_location=torch.device(device))
            net.load_state_dict(ckpoints)
            self.valid(data, net, data_dict, value_cnt)
        pickle.dump(data_dict, open('data_dict.pkl', 'wb'))
        data_dict = pickle.load(open('data_dict.pkl', 'rb'))
        with open('asimage.csv', 'w') as f:
            for i, path in enumerate(data_dict.keys()):
                if i == 0:
                    f.write(','.join(['', 'pic1' ,'aggressive', 'smart', 'trustworthy', 'beauty']) + '\n')
                item = data_dict[path]
                f.write(','.join([i, path, item[3], item[4], item[5], item[6]]) + '\n')
        with open('asperson.csv', 'w') as f:
            for i, path in enumerate(data_dict2.keys()):
                if i == 0:
                    f.write(','.join(['姓名', '父级文件夹' ,'aggressive', 'smart', 'trustworthy', 'beauty']) + '\n')
                item = data_dict[path]
                name, father_fn = path.split('/')[1], path.split('/')[0]
                tmp = []
                for p in item.keys():
                    tmp.append([data_dict[p][3], data_dict[p][4], data_dict[p][5], data_dict[p][6]])
                tmp = np.mean(np.array(tmp), axis=0)
                f.write(','.join([name, father_fn, tmp[0], tmp[1], tmp[2],tmp[3]]) + '\n')

    def test2(self):
        data_dict = {}
        data_dict2 = {}
        data_list = []
        root_dir = './faces'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    full_path = os.path.join(subdir, file)
                    parts = subdir.split('/')

                    img = Image.open(full_path).convert('RGB')
                    if transform is not None:
                        img = transform(img)
                    data_list.append([img, full_path])
                    data_dict[full_path] = [parts[-2], parts[-1], file, 0, 0, 0, 0]
                    if parts[-2]+'/'+parts[-1] not in data_dict2:
                        data_dict2[parts[-2]+'/'+parts[-1]] = {full_path:[0,0,0,0]}
                    else:
                        data_dict2[parts[-2]+'/'+parts[-1]][full_path] = [0,0,0,0]
        data_dict = pickle.load(open('data_dict.pkl', 'rb'))
        with open('asimage.csv', 'w') as f:
            for i, path in enumerate(data_dict.keys()):
                if i == 0:
                    f.write(','.join([' ', 'pic1' ,'aggressive', 'smart', 'trustworthy', 'beauty']) + '\n')
                item = data_dict[path]
                f.write(','.join([str(i), path, str(item[3]), str(item[4]), str(item[5]), str(item[6])]) + '\n')

        with open('asperson.csv', 'w') as f:
            for i, path in enumerate(data_dict2.keys()):
                if i == 0:
                    f.write(','.join(['姓名', '父级文件夹' ,'aggressive', 'smart', 'trustworthy', 'beauty']) + '\n')
                item = data_dict2[path]
                name, father_fn = path.split('/')[1], path.split('/')[0]
                tmp = []
                for p in item.keys():
                    tmp.append([data_dict[p][3], data_dict[p][4], data_dict[p][5], data_dict[p][6]])
                tmp = np.mean(np.array(tmp), axis=0)
                f.write(','.join([name, father_fn, str(tmp[0]), str(tmp[1]), str(tmp[2]), str(tmp[3])]) + '\n')
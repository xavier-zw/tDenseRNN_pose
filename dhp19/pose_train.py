# wu
# https://github.com/HowieMa/lstm_pm_pytorch.git
import argparse
# from model.lstm_pm import LSTM_PM
from DHP19Data import Dhp19PoseDataset
from hrnet import get_pose_net
from utils import JointsMSELoss,get_optimizer,save_loss
from config import config
from utils import AverageMeter, accuracy,get_final_preds
# from src.tools import *
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_ids = [0, 1, 2, 3]
import warnings
warnings.filterwarnings('ignore')
# hyper parameter
temporal = config.temporal
dataN = 2
if dataN == 1:
    train_data_dir = '../data/train_one/'
    train_label_dir = '../data/train_one/'

if dataN == 2:
    train_data_dir = '/home/shao_old/lance/实验/data/train'
    train_label_dir = '/home/shao_old/lance/实验/data/train'

if dataN == 3:
    train_data_dir = '../data/train_action/'
    train_label_dir = '../data/train_action/'



learning_rate = 8e-6
batch_size = 32
epochs = 100
begin_epoch = 0
save_dir = './ckpt2/'
cuda = 1

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
train_data = Dhp19PoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=temporal, train=True)
print('Train dataset total number of images sequence is ----' + str(len(train_data)))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, num_workers=32, shuffle=True, pin_memory=True, drop_last=True)

net = get_pose_net()
model = torch.nn.DataParallel(net, device_ids=device_ids).cuda()


def load_model(model):
    # build model
    net = get_pose_net()
    if torch.cuda.is_available():
        net = net.cuda()
        # net = nn.DataParallel(net)  # multi-Gpu

    save_path = os.path.join('ckpt2/tdense' + str(model)+'.pth')
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)
    return net

# begin_epoch = 0
# net = load_model(24)
# model = torch.nn.DataParallel(net).cuda()


def train():
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.1)

    # criterion = nn.MSELoss(size_average=True)
    for epoch in range(begin_epoch, epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        model.train()
        end = time.time()
        with tqdm(total=len(train_dataset), leave=True,desc="epoch"+str(epoch),ncols=100, unit='it', unit_scale=True) as t:
            for i, (inputs, targets, target_weights) in enumerate(train_dataset):
                outputs = model(inputs.cuda())
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.update(loss.item(), inputs.size(0))
                for j in range(temporal):
                    out = outputs#[:, j]
                    tar = targets[:, j]
                    _, avg_acc, cnt, pred = accuracy(out.detach().cpu().numpy(),
                                                     tar.detach().cpu().numpy())
                    acc.update(avg_acc, cnt)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                t.update()
                t.set_postfix(lr=optimizer.param_groups[0]["lr"], loss=losses.avg, acc=acc.avg)
        mse_dist = val()
        lr_scheduler.step(mse_dist)
        if epoch % 1 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'ucihand_lstm_pm{:d}.pth'.format(epoch)))
        if optimizer.param_groups[0]["lr"] < 1e-6:
            break
    print('train done!')
def val():

    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()
        path_ = '/home/shao_old/lance/实验/data_test/select'
        path_list = os.listdir(path_)
        path_list.sort(key=lambda x: (int(x[0:1]), int(x[2:3])))
        model.eval()
        res_out = 0
        for ac in path_list:
            val_data_dir = os.path.join(path_, ac)
            val_label_dir = os.path.join(path_, ac)
            val_data = Dhp19PoseDataset(data_dir=val_data_dir, label_dir=val_label_dir,
                                                temporal=config.temporal,
                                                train=False)
                # print('Train dataset total number of images sequence is ----' + str(len(train_data)))

                # Data Loader
            val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=32,
                                     drop_last=True)

            res = 0
            num = 0
            for i, (inputs, targets, target_weight, cs, ss, joints, data_numpy, name) in enumerate(val_dataset):
                # print(input.shape) # (batch,1,256,192)
                # print(name)
                num += 1
                outputs = model(inputs.cuda())
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                res_a = 0
                for j in range(config.temporal):
                    out = outputs#[:, j]
                    tar = targets[:, j]
                    joint = joints[:, j]
                    c = cs[:, j]
                    s = ss[:, j]
                    _, avg_acc, cnt, pred = accuracy(out.cpu().numpy(), tar.cpu().numpy())
                    acc.update(avg_acc, cnt)
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    preds, maxvals = get_final_preds(
                        config, out.clone().cpu().numpy(), c.numpy(), s.numpy())
                    joint = np.array(joint)
                    preds = np.array(preds)
                    dist_2d = np.linalg.norm((joint[:, :, :2] - preds), axis=-1)
                    res_a += np.nanmean(dist_2d)
                    # print((res_a))
                res += res_a / config.temporal
                # print(res / num)
            res_out += res / num
            # print(ac+":"+str(res/num))
        # print(res_out)
    print("mpjpe:", str(res_out / len(path_list)), "|| acc: %s" % str(acc.avg))
    return res_out / len(path_list)
if __name__ == '__main__':
    train()









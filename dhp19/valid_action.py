# https://github.com/HowieMa/lstm_pm_pytorch.git
import argparse
# from model.lstm_pm import LSTM_PM
from DHP19Data import Dhp19PoseDataset
from tDense import get_pose_net
from utils import JointsMSELoss,get_optimizer,save_loss
from config import config
from utils import AverageMeter, accuracy,get_final_preds
# from src.tools import *
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import math
from config import config
device_ids = [0]

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


learning_rate = 8e-6
batch_size = 4
epochs = 50
begin_epoch = 0
save_dir = './ckpt2/'
cuda = 1

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset


net = get_pose_net()
gpus = [int(i) for i in '0'.split(',')]
final_output_dir = "ckpt2/ucihand_lstm_pm50.pth"

def load_model(model):
    # build model
    net = get_pose_net()
    if torch.cuda.is_available():
        net = net.cuda()
        # net = nn.DataParallel(net)  # multi-Gpu

    save_path = os.path.join('ckpt2/ucihand_lstm_pm' + str(model)+'.pth')
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)
    return net



import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
def plot_2d(dvs_frame, joint, pre):
    " To plot image and 2D ground truth and prediction "
    # plt.figure()
    plt.cla()
    # plt.imshow(dvs_frame)
    plt.imshow(np.zeros((300,300)))

    # joint = np.array(joint)
    # pre = np.array(pre)
    plt.plot(joint[:, 0], joint[:, 1], '.', c='red', label='gt')
    plt.plot(pre[:, 0], pre[:, 1], '.', c='blue', label='gt')
    plt.show()
    plt.pause(0.00002)

plt.ion()
# list = [7]
list = list(range(0,100))
# list = [6]
# list = [5,10,15,20,25,30,35,40,45,50]
def train():
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()



    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode


    with torch.no_grad():
        end = time.time()
        path_ = '/home/shao/lance/实验/data_test/select'
        path_list = os.listdir(path_)
        path_list.sort(key=lambda x: (int(x[0:1]), int(x[2:3])))

        for l in list:
            net = load_model(l)
            model = torch.nn.DataParallel(net, device_ids=gpus).cuda()
            model.eval()
            res_out = 0
            for ac in path_list:
                train_data_dir = os.path.join(path_,ac)
                train_label_dir = os.path.join(path_,ac)
                train_data = Dhp19PoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=config.temporal,
                                              train=False)
                # print('Train dataset total number of images sequence is ----' + str(len(train_data)))

                # Data Loader
                train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=False)

                res = 0
                num = 0
                for i, (inputs, targets, target_weight, cs, ss, joints, data_numpy, name) in enumerate(train_dataset):
                    # print(input.shape) # (batch,1,256,192)
                    # print(name)
                    num+=1
                    outputs = model(inputs.cuda())
                    targets = targets.cuda()
                    loss = criterion(outputs, targets)
                    losses.update(loss.item(), inputs.size(0))
                    res_a = 0
                    for j in range(config.temporal):
                        out = outputs[j]
                        tar = targets[:, j]
                        joint = joints[:, j]
                        c = cs[:, j]
                        s = ss[:, j]

                        _, avg_acc, cnt, pred = accuracy(out.cpu().numpy(),
                                                         tar.cpu().numpy())
                        acc.update(avg_acc, cnt)
                        # measure elapsed time
                        batch_time.update(time.time() - end)
                        end = time.time()
                        preds, maxvals = get_final_preds(
                            config, out.clone().cpu().numpy(), c.numpy(), s.numpy())
                        mpjpe = 0
                        for i in range(data_numpy.shape[0]):
                            joint = np.array(joint)
                            preds = np.array(preds)
                            # for j in range(13):
                            dist_2d = np.linalg.norm((joint[:,:,:2] - preds), axis=-1)
                            mpjpe += np.nanmean(dist_2d)
                                # res += math.fabs(joint[i][j][0] - preds[i][j][0]) + math.fabs(joint[i][j][1] - preds[i][j][1])
                            # plot_2d(data_numpy[i], joint[i], preds[i])
                        # print(res//(13))
                        res_a += mpjpe/data_numpy.shape[0]
                        # print((res_a))
                    res += res_a/config.temporal
                    # print(res / num)
                res_out += res/num
                # print(ac+":"+str(res/num))
            # print(res_out)
            print(str(l)+":"+str(res_out/33))
    print('train done!')

if __name__ == '__main__':
    train()









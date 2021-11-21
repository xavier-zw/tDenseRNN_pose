from torch.utils.data import Dataset
import os
import json_tricks as json
import numpy as np
from utils import get_affine_transform, affine_transform
import cv2
import matplotlib.image as mpimg
import torch
import copy
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import config
import scipy.misc
from config import config


class Dhp19PoseDataset(Dataset):
    def __init__(self, data_dir, label_dir, train, temporal=5, joints=13, transform=None, sigma=1):
        self.seqs = os.listdir(data_dir)
        self.data_dir = data_dir
        self.temporal = config.temporal
        self.temporal_dir = []
        self.labels = []
        self.label_dir = label_dir
        self.image_size = [256, 256]
        self.heatmap_size = [128, 128]
        self.size = 0
        self.train = train
        self.gen_temporal_dir()

    def plot_2d(self, dvs_frame, joint):
        " To plot image and 2D ground truth and prediction "
        # plt.figure()
        plt.cla()
        plt.imshow(dvs_frame)

        temp = np.array(joint)
        plt.plot(temp[:, 0], temp[:, 1], '.', c='red', label='gt')
        plt.show()
        plt.pause(0.0002)

    plt.ion()

    def show(self):
        print(len(self.temporal_dir))
        for idx in range(len(self.temporal_dir)):
            img_path = self.temporal_dir[idx]
            print(img_path)
            joint = self.labels[idx]
            data_numpy = mpimg.imread(img_path)
            joint = np.array(joint)
            self.plot_2d(data_numpy, joint)

    def gen_temporal_dir(self):
        self.seqs.sort(key=lambda x: (int(x[:2]), x[3:4], x[5:6]))
        for seq in self.seqs:
            if seq == 'annot':
                continue
            image_path = os.path.join(self.data_dir, seq)
            imgs = os.listdir(image_path + '/images/')
            imgs.sort(
                key=lambda x: (int(x[x.index('_', 6) + 1:x.index('.')])))
            img_num = len(imgs)
            label_path = os.path.join(self.label_dir, seq)
            lables = json.load(open(label_path + '/annot/' + seq + '.json'))

            for i in range(img_num // self.temporal):
                tmp1 = []
                tmp2 = []
                for k in range(self.temporal * i, self.temporal * (i + 1)):
                    tmp1.append(os.path.join(os.path.join(image_path + '/images', imgs[k])))
                    tmp2.append(lables[str(k)])
                self.temporal_dir.append(tmp1)
                self.labels.append(tmp2)
        self.joint_true = self.labels
        # self.show()
        # print(len(self.temporal_dir))
        # print('total numbers of image sequence is ' + str(len(self.temporal_dir)))

    def __len__(self, ):
        return len(self.temporal_dir)

    def __getitem__(self, idx):
        inputs = torch.zeros((self.temporal, 1, self.image_size[0], self.image_size[1]))
        targets = torch.zeros((self.temporal, 13, self.heatmap_size[0], self.heatmap_size[1]))
        target_weights = torch.zeros((self.temporal, 13, 3))
        cs = torch.zeros(self.temporal, 2)
        ss = torch.zeros(self.temporal, 2)
        joints_ture = torch.zeros(self.temporal, 13, 3)

        for k in range(self.temporal):
            img_path = self.temporal_dir[idx][k]
            joint = self.labels[idx][k]
            joint_true = self.joint_true[idx][k]
            data_numpy = mpimg.imread(img_path)
            # self.plot_2d(data_numpy, joint)
            joint = np.array(joint)
            joint_true = np.array(joint_true)
            u = joint[:, 0]
            v = joint[:, 1]
            c = np.array([(max(u) + min(u)) / 2, (max(v) + min(v)) / 2], dtype=np.float)
            c = np.array(c, dtype=np.float)
            s = (max(v) - min(v)) * 0.0065
            s = np.array([s, s], dtype=np.float)
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25
            c = c - 1
            r = 0
            cs[k] = torch.from_numpy(c)
            ss[k] = torch.from_numpy(s)
            joints_ture[k] = torch.from_numpy(joint_true)
            trans = get_affine_transform(c, s, r, self.image_size)
            input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            for i in range(13):
                # if joints_vis[i, 0] > 0.0:
                joint[i, 0:2] = affine_transform(joint[i, 0:2], trans)
            # print(joints_vis)
            target, target_weight = self.generate_target(joint, joint)
            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)
            aa = torch.zeros((1, 256, 256))  # simple
            aa[0, :, :] = torch.from_numpy(input)
            input = aa
            inputs[k] = input
            targets[k] = target
            target_weights[k] = target_weight
        if self.train:
            return inputs, targets, target_weights
        else:
            return inputs, targets, target_weights, cs, ss, joints_ture, data_numpy, self.temporal_dir[idx]

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((13, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((13,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = 2 * 3

        for joint_id in range(13):
            feat_stride = [a / b for a, b in zip(self.image_size, self.heatmap_size)]
            # feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * 2 ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight

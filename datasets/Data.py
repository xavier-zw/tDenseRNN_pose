import PIL.Image
from torch.utils.data import Dataset
import os
import numpy as np
from tools.utils import get_affine_transform,affine_transform
import torch
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import albumentations as A

class CDES25Dataset(Dataset):
    def __init__(self, data_dir_list, label_dir_list, config, train):
        self.data_dir_list = data_dir_list
        self.CFG = config
        self.temporal = config.temporal
        self.transform = config.transform
        self.temporal_dir = []
        self.labels = []
        self.label_dir_list = label_dir_list
        self.image_size = config.IMAGE_SIZE
        self.heatmap_size =config.HEATMAP_SIZE
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
            joint = self.labels[idx]
            data_numpy = mpimg.imread(img_path)
            joint = np.array(joint)
            self.plot_2d(data_numpy,joint)

    def read(self, path):
        res = torch.zeros(self.CFG.NUM_JOINTS, 2)
        with open(path) as f:
            data = f.read().split("\n")
        if data[0] == "13":
            for index,x in enumerate(data[2:-1]):
                x = x.split(" ")
                res[index][0] = round(float(x[0][1:-1]))
                res[index][1] = round(float(x[1][1:-1]))
        return res
    def gen_temporal_dir(self):
        for data_dir_name,label_dir_name in zip(self.data_dir_list,self.label_dir_list):
            label_image_name_list = sorted(os.listdir(label_dir_name))
            label_num = len(label_image_name_list)
            for i in range(label_num // self.temporal):
                tmp1 = []
                tmp2 = []
                try:
                    for k in range(self.temporal * i, self.temporal * (i + 1)):
                        imgs = os.path.join(data_dir_name,label_image_name_list[k][:-4]+".png")
                        lables = os.path.join(label_dir_name, label_image_name_list[k])
                        lables = self.read(lables)
                        tmp1.append(imgs)
                        tmp2.append(lables)
                    self.temporal_dir.append(tmp1)
                    self.labels.append(tmp2)
                except:
                    pass
        # self.joint_true = self.labels
    def __len__(self,):
        return len(self.temporal_dir)
    def __getitem__(self, idx):
        inputs = torch.zeros((self.temporal, 1, self.image_size[0], self.image_size[1]))
        targets = torch.zeros((self.temporal, self.CFG.NUM_JOINTS, self.heatmap_size[0],self.heatmap_size[1]))
        target_weights = torch.zeros((self.temporal, self.CFG.NUM_JOINTS, 2))
        min_u, min_v, max_u, max_v = 9999, 9999, 0, 0
        for k in range(self.temporal):
            img_path = self.temporal_dir[idx][k]
            joint = self.labels[idx][k]
            joint = np.array(joint)
            u = joint[:, 0]
            v = joint[:, 1]
            min_u = min(min(u), min_u)
            max_u = max(max(u), max_u)
            min_v = min(min(v), min_v)
            max_v = max(max(v), max_v)
        min_v = min_v - (100 if min_v > 100 else min_v)
        min_u = min_u - (100 if min_u > 100 else min_u)
        max_v = max_v + (100 if max_v < self.CFG.SOURCE_SIZE[1] else self.CFG.SOURCE_SIZE[1] - max_v)
        max_u = max_u + (100 if max_u < self.CFG.SOURCE_SIZE[0] else self.CFG.SOURCE_SIZE[0] - max_u)
        degree = np.random.randint(0, 4)
        for k in range(self.temporal):
            img_path = self.temporal_dir[idx][k]
            joint = self.labels[idx][k]
            data_image = Image.open(img_path).convert('L')
            joint = np.array(joint)

            joint[:, 0] = (joint[:, 0] - min_u) * (self.CFG.IMAGE_SIZE[0]/(max_u-min_u))
            joint[:, 1] = (joint[:, 1] - min_v) * (self.CFG.IMAGE_SIZE[1]/(max_v-min_v))

            input = np.array(data_image)[int(min_v):int(max_v), int(min_u):int(max_u)]
            input = self.transform(PIL.Image.fromarray(input))

            target, target_weight = self.generate_target(joint, joint)
            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)

            if self.train:
                input = self.dumpRotateImage(np.array(input[0]), degree*90)
                input = torch.from_numpy(input)
                for i in range(target.shape[0]):
                    target_i = self.dumpRotateImage(np.array(target[i]), degree*90)
                    target[i] = torch.from_numpy(target_i)

                # fig,ax = plt.subplots(1)
                # ax.imshow(input, cmap=plt.cm.bone)
                # ax.imshow(np.array(Image.fromarray(np.array(torch.sum(target,dim=0))).resize((256,256))), alpha=0.3, cmap="Blues")
                # plt.show()

            inputs[k] = input
            targets[k] = target
            target_weights[k] = target_weight
        # fig, ax = plt.subplots(1, self.CFG.temporal)
        # for i in range(self.CFG.temporal):
        #     ax[i].imshow(inputs[i][0], cmap=plt.cm.bone)
        #     ax[i].imshow(np.array(Image.fromarray(np.array(torch.sum(targets[i],dim=0))).resize((256,256))), alpha=0.3, cmap="Blues")
        # plt.show()

        return inputs, targets, target_weights



    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.CFG.NUM_JOINTS, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]


        target = np.zeros((self.CFG.NUM_JOINTS,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = 2 * 3

        for joint_id in range(self.CFG.NUM_JOINTS):
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


    def dumpRotateImage(self, img, degree):
        height, width = img.shape[:2]
        heightNew = int(width * np.fabs(np.sin(np.radians(degree))) + height * np.fabs(np.cos(np.radians(degree))))
        widthNew = int(height * np.fabs(np.sin(np.radians(degree))) + width * np.fabs(np.cos(np.radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2

        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0))

        return imgRotation



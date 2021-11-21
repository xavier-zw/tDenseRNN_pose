import models
from tools.config import CFG
from tools.utils import AverageMeter, accuracy, get_final_preds
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tools.utils import generate_target
import argparse


def read(config, path):
    res = torch.zeros(config.NUM_JOINTS, 2)
    with open(path) as f:
        data = f.read().split("\n")
    if data[0] == "13":
        for index, x in enumerate(data[2:-1]):
            x = x.split(" ")
            res[index][0] = round(float(x[0][1:-1]))
            res[index][1] = round(float(x[1][1:-1]))
    return res


def load_model(model_path, model_name, cuda=True):
    model = eval('models.' + model_name + '.get_pose_net')(CFG)
    model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    return model


def plot_2d(CFG, dvs_frame, joint, pre, index, save_path):
    " To plot image and 2D ground truth and prediction "
    # plt.figure()
    # fig, ax = plt.subplots(1, CFG.temporal, figsize=(50, 50))
    # for i in range(CFG.temporal):
    #     ax[i].imshow(dvs_frame[i])
    #     ax[i].plot(joint[0, i, :, 0], joint[0, i, :, 1], '.', c='red', label='gt')
    #     ax[i].plot(pre[i][0, :, 0], pre[i][0, :, 1], '.', c='blue', label='pred')
    # plt.savefig("show/%s.png"%str(index))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    skelenton = [[0, 1], [1, 3], [3, 5], [1, 7], [1, 2], [7, 9], [9, 11],
                 [0, 2], [2, 4], [4, 6], [2, 8], [7, 8], [8, 10], [10, 12]]
    for i in range(CFG.temporal):
        plt.cla()
        plt.axis('off')
        plt.imshow(dvs_frame[i])
        # plt.plot(joint[0, i, :, 0], joint[0, i, :, 1], '.', c='red', label='gt')
        # plt.plot(pre[i][0, :, 0], pre[i][0, :, 1], '.', c='blue', label='gt')
        for s_k in skelenton:
            plt.plot(pre[i][0, s_k, 0], pre[i][0, s_k, 1], linewidth=3)
        plt.savefig(save_path+"/%s.png" % (str(index) + "_" + str(i)), bbox_inches='tight')


def valid(CFG, inputs, targets, joint_true, image_true, rectangle, i, save_path):

    inputs, targets,joint_true = inputs.unsqueeze(dim=0), targets.unsqueeze(dim=0), joint_true.unsqueeze(dim=0)
    pred_all = []
    with torch.no_grad():
        outputs = model(inputs.cuda())
    targets = targets.cuda()
    for j in range(CFG.temporal):
        out = outputs[:, j]
        tar = targets[:, j]
        joint = joint_true[:, j]
        _, avg_acc, cnt, pred, oks = accuracy(out.cpu().numpy(),
                                              tar.cpu().numpy())
        preds, maxvals = get_final_preds(out.clone().cpu().numpy(),rectangle)
        pred_all.append(preds)
    plot_2d(CFG, image_true, joint_true, pred_all, i, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--modelName',
                        help='model names (tDense or other)',
                        type=str,
                        default="tDense"
    )
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default="./valid_model/tDense.pth")
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default="./valid_data/A0017P0005/S00")
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    learning_rate = CFG.LR
    batch_size = CFG.BATCH_SIZE
    epochs = CFG.END_EPOCH
    begin_epoch = CFG.BEGIN_EPOCH
    save_dir = CFG.SAVE_PATH

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    transform = CFG.transform

    net = load_model(args.modelDir, args.modelName)
    model = net.cuda()
    model.eval()
    valid_path = args.dataDir
    valid_data_list = sorted(os.listdir(os.path.join(valid_path, "image_event_binary")))
    valid_data_label = sorted(os.listdir(os.path.join(valid_path, "label_event_fill")))
    length = len(valid_data_label) // CFG.temporal

    for i in range(length):
        temporal_image_name = valid_data_label[i * CFG.temporal:(i + 1) * CFG.temporal]
        labels = []
        inputs = torch.zeros((CFG.temporal, 1, CFG.IMAGE_SIZE[0], CFG.IMAGE_SIZE[1]))
        targets = torch.zeros((CFG.temporal, CFG.NUM_JOINTS, CFG.HEATMAP_SIZE[0], CFG.HEATMAP_SIZE[1]))
        target_weights = torch.zeros((CFG.temporal, CFG.NUM_JOINTS, 3))
        joint_true = torch.zeros((CFG.temporal, CFG.NUM_JOINTS, 2))
        min_u, min_v, max_u, max_v = 9999, 9999, 0, 0
        for k in range(CFG.temporal):
            joint = read(CFG, os.path.join(valid_path, "label_event_fill", temporal_image_name[k]))
            joint = np.array(joint)
            joint_true[k] = torch.from_numpy(joint)
            u = joint[:, 0]
            v = joint[:, 1]
            min_u = min(min(u), min_u)
            max_u = max(max(u), max_u)
            min_v = min(min(v), min_v)
            max_v = max(max(v), max_v)
        min_v = min_v - (100 if min_v > 100 else min_v)
        min_u = min_u - (100 if min_u > 100 else min_u)
        max_v = max_v + (100 if max_v < CFG.SOURCE_SIZE[1] else CFG.SOURCE_SIZE[1] - max_v)
        max_u = max_u + (100 if max_u < CFG.SOURCE_SIZE[0] else CFG.SOURCE_SIZE[0] - max_u)
        rectangle = [min_u, max_u, min_v, max_v]
        image_true = torch.zeros((CFG.temporal, CFG.SOURCE_SIZE[1], CFG.SOURCE_SIZE[0]))
        for k in range(CFG.temporal):
            img_path = os.path.join(valid_path, "image_event_binary", temporal_image_name[k][:-4] + ".png")
            joint = read(CFG, os.path.join(valid_path, "label_event_fill", temporal_image_name[k]))
            data_image = Image.open(img_path).convert('L')
            image_true[k] = torch.from_numpy(np.array(data_image))
            # self.plot_2d(data_numpy, joint)
            joint = np.array(joint)
            joint[:, 0] = (joint[:, 0] - min_u) * (CFG.IMAGE_SIZE[0] / (max_u - min_u))
            joint[:, 1] = (joint[:, 1] - min_v) * (CFG.IMAGE_SIZE[1] / (max_v - min_v))
            input = np.array(data_image)[int(min_v):int(max_v), int(min_u):int(max_u)]
            input = transform(Image.fromarray(input))
            target, target_weight = generate_target(CFG, joint, joint)
            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)
            inputs[k] = input
            targets[k] = target
            target_weights[k] = target_weight

        valid(CFG, inputs, targets, joint_true, image_true, rectangle, i, CFG.save_show_image)
from datasets.Data import CDES25Dataset
from tools.utils import JointsMSELoss, get_optimizer,save_loss
from tools.config import CFG
from tools.utils import AverageMeter, accuracy, get_final_preds
import os
import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import models
import datasets

def load_model(model_path, model_name, cuda=True):
    model = eval('models.' + model_name + '.get_pose_net')(CFG)
    model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    return model

def val():
    batch_time = AverageMeter()
    acc_pck = AverageMeter()
    acc_oks = AverageMeter()
    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()
        model.eval()
        num = 0
        oks_res = []
        for i, (inputs, targets, target_weight) in enumerate(valid_dataset):
            num += 1
            outputs = model(inputs.to(device))
            targets = targets.to(device)
            for j in range(CFG.temporal):
                out = outputs[:, j]
                tar = targets[:, j]
                _, avg_acc, cnt, pred, oks = accuracy(out.cpu().numpy(), tar.cpu().numpy())
                oks_res += oks
                acc_pck.update(avg_acc, cnt)
                acc_oks.update(np.array(oks).mean(), cnt)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    ap_5 = np.sum(np.array(oks_res) >= 0.5) / len(oks_res)
    ap_75 = np.sum(np.array(oks_res) >= 0.75) / len(oks_res)
    with open(CFG.result_txt, "a", encoding="utf-8") as f:
        f.write("pck:" + str(acc_pck.avg) + "\nAP:" + str(acc_oks.avg) + "\nAP_0.5:" + str(ap_5) + "\nAP_0.75:" + str(ap_75) + "\n" + "*"*50 + "\n")
    print("pck:",str(acc_pck.avg),"\nAP:",str(acc_oks.avg),"\nAP_0.5:",str(ap_5),"\nAP_0.75:",str(ap_75))
    res = acc_oks.avg
    acc_pck.reset()
    acc_oks.reset()
    return res


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
                        default="./data/sample_valid")
    parser.add_argument('--batch_size',
                        help='data directory',
                        type=int,
                        default=4)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    device = CFG.device
    # hyper parameter
    temporal = CFG.temporal

    root_valid = args.dataDir
    val_data_dir_list = os.listdir(root_valid)
    val_train_data_list = []
    val_train_data_label = []
    print("Start creat dir!")
    for x in tqdm(val_data_dir_list):
        val_train_data_list.append(os.path.join(root_valid, x, "S00", "image_event_binary"))
        val_train_data_label.append(os.path.join(root_valid, x, "S00", "label_event_fill"))
    print("Done!")

    batch_size = args.batch_size

    # Build dataset
    valid_data = CDES25Dataset(val_train_data_list, val_train_data_label, CFG, False)

    # Data Loader
    valid_dataset = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    net = load_model(args.modelDir, args.modelName)
    model = net.cuda()
    val()

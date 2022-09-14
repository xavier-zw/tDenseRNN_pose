from datasets.Data import CDES25Dataset
from datasets.JHMDB import JHMDB_Data
from datasets.CDEHP_Event import CDEHP_Event
from datasets.Test_AUG import Test_AUG
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
import random
import models
import datasets
from tools.utils import get_max_preds


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(path):
    # build models
    net = eval('models.' + CFG.MODEL_NAME + '.get_pose_net')(CFG)
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    return net


def train():
    for epoch in range(begin_epoch, epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc_pck = AverageMeter()
        acc_oks = AverageMeter()
        model.train()
        end = time.time()
        with tqdm(total=len(train_dataset), leave=True, desc="epoch" + str(epoch), ncols=150, unit='it',
                  unit_scale=True) as t:
            for i, (inputs, targets, target_weights) in enumerate(train_dataset):
                targets = targets.to(device)
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), inputs.size(0))
                for j in range(CFG.temporal):
                    out = outputs[:, j]
                    tar = targets[:, j]
                    _, avg_acc, cnt, pred, oks, _ = accuracy(out.detach().cpu().numpy(), tar.detach().cpu().numpy())
                    acc_pck.update(avg_acc, cnt)
                    acc_oks.update(np.array(oks).mean(), cnt)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                t.update()
                t.set_postfix(loss=losses.avg, acc_pck=acc_pck.avg, acc_oks=acc_oks.avg, lr=optimizer.param_groups[0]["lr"])
        val_acc = val()
        lr_scheduler.step(val_acc)
        if optimizer.param_groups[0]["lr"] < 1e-6:
            break
        if epoch % 1 == 0:
            torch.save(model.module.state_dict(), os.path.join(save_dir, 'model_{:d}.pth'.format(epoch)),_use_new_zipfile_serialization=False)
    print('train done!')


def val():
    batch_time = AverageMeter()
    acc_pck = AverageMeter()
    acc_oks = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        num = 0
        oks_res = []
        for i, (inputs, targets, target_weight) in enumerate(valid_dataset):
            num += 1
            targets = targets.to(device)
            outputs = model(inputs.to(device))
            for j in range(CFG.temporal):
                out = outputs[:, j]
                tar = targets[:, j]
                _, avg_acc, cnt, pred, oks, _ = accuracy(out.cpu().numpy(), tar.cpu().numpy())
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


if __name__ == '__main__':
    setup_seed(999)
    device = CFG.device
    # hyper parameter
    temporal = CFG.temporal
    root_train = CFG.ROOT_TRAIN
    train_data_dir_list = sorted(os.listdir(root_train))
    train_data_list = []

    train_data_label = []
    root_valid = CFG.ROOT_VALID
    val_data_dir_list = sorted(os.listdir(root_valid))
    val_train_data_list = []
    val_train_data_label = []
    print("Start creat dir!")
    for x in tqdm(train_data_dir_list):
        train_data_list.append(os.path.join(root_train, x, "S00", "vector_event_binary"))
        train_data_label.append(os.path.join(root_train, x, "S00", "label_event_fill"))
    for x in tqdm(val_data_dir_list):
        val_train_data_list.append(os.path.join(root_valid, x, "S00", "vector_event_binary"))
        val_train_data_label.append(os.path.join(root_valid, x, "S00", "label_event_fill"))
    print("Done!")
    learning_rate = CFG.LR
    batch_size = CFG.BATCH_SIZE
    epochs = CFG.END_EPOCH
    begin_epoch = CFG.BEGIN_EPOCH
    save_dir = CFG.SAVE_PATH
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Build dataset
    train_data = Test_AUG(train_data_list, train_data_label, CFG, True)
    valid_data = Test_AUG(val_train_data_list, val_train_data_label, CFG, False)
    print('Train dataset total number of images sequence is ----' + str(len(train_data)))

    # Data Loader
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    valid_dataset = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    model = eval('models.' + CFG.MODEL_NAME + '.get_pose_net')(CFG)
    # model = load_model("/home/xavier/Pycharm/tRnn_pose/QKV_Pose_nonlocal/model_9.pth")
    model = model.to(device)
    print(model)
    model = nn.DataParallel(model).to(device)
    criterion = JointsMSELoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.001)
    train()
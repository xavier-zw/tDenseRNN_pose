import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import copy
from config import config
image_h, image_w, num_joints = 260, 346, 13
image_size = [192, 256]
heatmap_size = [48, 64]

def loss_history_init(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history
def save_loss(predict_heatmaps, label_map, epoch, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save = loss_history_init(temporal=temporal)

    predict = predict_heatmaps[0]
    target = label_map[:, 0, :, :, :]
    initial_loss = criterion(predict, target)  # loss initial
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t + 1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)  # loss in each stage
        total_loss += tmp_loss
        loss_save['temporal' + str(t)] = float('%.8f' % tmp_loss)

    total_loss = total_loss
    loss_save['total'] = float(total_loss)

    # save loss to file
    # if train is True:
    #     if not os.path.exists(save_dir + 'loss_epoch' + str(epoch)):
    #         os.mkdir(save_dir + 'loss_epoch' + str(epoch))
    #     json.dump(loss_save, open(save_dir + 'loss_epoch' + str(epoch) + '/s' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))
    #
    # else:
    #     if not os.path.exists(save_dir + 'loss_test/'):
    #         os.mkdir(save_dir + 'loss_test/')
    #     json.dump(loss_save, open(save_dir + 'loss_test/' + str(step).zfill(4) + '.json', 'w', encoding="utf8"))

    return total_loss


def get_joints(x_selected, y_selected):
    vicon_xyz_homog = np.concatenate([y_selected, np.ones([1, 13])], axis=0)
    coord_pix_all_cam2_homog = np.matmul(np.load('P2.npy'), vicon_xyz_homog)
    coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog / coord_pix_all_cam2_homog[-1]

    u = coord_pix_all_cam2_homog_norm[0]
    v = image_h - coord_pix_all_cam2_homog_norm[1]  # flip v coordinate to match the image direction
    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[u > image_w] = 0
    mask[u <= 0] = 0
    mask[v > image_h] = 0
    mask[v <= 0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    joints_3d = np.zeros((13, 3), dtype=np.float)
    joints_3d_vis = np.ones((13, 3), dtype=np.int)
    joints_3d_vis[:, 1] = 1
    joints_3d[:, 0] = u
    joints_3d[:, 1] = v

    joints_3d[:, 0:2] = joints_3d[:, 0:2] - 1
    # u = joints_3d[:, 0]
    # v = joints_3d[:, 1]
    c = np.array([(max(u) + min(u)) / 2, (max(v) + min(v)) / 2], dtype=np.float)
    s = (max(v) - min(v)) * 0.0065
    s = np.array(s, dtype=np.float)
    c = np.array(c, dtype=np.float)
    s = np.array([s, s], dtype=np.float)
    r = 0

    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(
        x_selected,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    joints_bak = copy.deepcopy(joints_3d)
    for i in range(13):
        # if joints_3d_vis[i, 0] > 0.0:
        joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

    aa = np.zeros((1, 256, 192))  # simple
    aa[0, :, :] = input
    input = aa
    return c, s, input, joints_bak
def predict_DHP(model, x_selected, y_selected, ch_idx, imgidx):
    c = np.zeros((x_selected.shape[0],2))
    s = np.zeros((x_selected.shape[0],2))
    joints_3d = np.zeros((x_selected.shape[0],13,3))
    input = np.zeros((x_selected.shape[0],1,256,192))
    for i in range(x_selected.shape[0]):
        c[i,:],s[i,:],input[i,:,:,:],joints_3d[i,:,:]= get_joints(x_selected[i],y_selected[i])

    input = torch.from_numpy(input)
    output = model(input.float())
    preds, maxvals = get_final_preds(
        config, output.clone().detach().cpu().numpy(), c, s)
    dist_2d = np.linalg.norm((joints_3d[:,:,:2] - preds), axis=-1)
    mpjpe = np.nanmean(dist_2d)
    for j in range(preds.shape[0]):
        plt.cla()
        # plt.imshow(input[j,0,:,:])
        plt.imshow(x_selected[j,:,:])
        plt.plot(joints_3d[j,:,0], joints_3d[j,:,1], '.', c='red', label='gt')
        plt.plot(preds[j,:,0], preds[j,:,1], '.', c='blue', label='gt')
        plt.show()
        plt.pause(0.5)
    return mpjpe


def generate_target(joints, joints_vis):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    target_weight = np.ones((13, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]



    target = np.zeros((13,
                       48,
                       64),
                      dtype=np.float32)

    tmp_size = 2 * 3

    for joint_id in range(13):
        feat_stride = [a / b for a, b in zip(image_size, heatmap_size)]
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
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
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return target, target_weight
def valide(batch, data_numpy, preds, joints_bak):
    size = 200
    mejpe_list = []
    for j in range(preds.shape[0]):
        # plt.cla()
        # plt.imshow(data_numpy[j])
        # plt.plot(joints_bak[j][:, 0], joints_bak[j][:, 1], '.', c='red', label='gt')
        # plt.plot(preds[j][:, 0], preds[j][:, 1], '.', c='blue', label='gt')
        # plt.show()
        mejpe = np.nanmean(np.linalg.norm((joints_bak[j][:, :2] - preds[j]), axis=-1))
        mejpe_list.append(mejpe)
        # print(mejpe)
        # plt.pause(0.1)
    x = np.arange(len(mejpe_list))
    if batch%size == 0:
        plt.cla()
        plt.scatter(x, mejpe_list)
        plt.plot(x, mejpe_list, 'r-', lw=5)
        plt.show()
        plt.pause(0.5)
        mejpe_list = []

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords
def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    # if config.TEST.POST_PROCESS:
    #     for n in range(coords.shape[0]):
    #         for p in range(coords.shape[1]):
    #             hm = batch_heatmaps[n][p]
    #             px = int(math.floor(coords[n][p][0] + 0.5))
    #             py = int(math.floor(coords[n][p][1] + 0.5))
    #             if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
    #                 diff = np.array([hm[py][px+1] - hm[py][px-1],
    #                                  hm[py+1][px]-hm[py-1][px]])
    #                 coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals
def plot_2d(dvs_frame, sample_gt):
    import matplotlib;matplotlib.use("TKAgg")
    import matplotlib.pyplot as plt
    plt.imshow(dvs_frame)
    plt.plot(sample_gt[:, 0], sample_gt[:, 1], '.', c='red', label='gt')
    # plt.plot(sample_pred[:, 0], sample_pred[:, 1], '.', c='blue', label='pred')
    # plt.legend()
    plt.show()
def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists
def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1
def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        # print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rms':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )


    return optimizer

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, outputs, targets):
        total_loss = self.criterion(outputs, targets)
        return total_loss
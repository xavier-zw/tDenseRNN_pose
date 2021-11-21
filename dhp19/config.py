from easydict import EasyDict as edict

config = edict()
config.temporal = 1
config.GPUS = '0'
config.OUTPUT_DIR = ''
config.WORKERS= 0
config.PRINT_FREQ = 20
config.TRAIN = edict()
config.TRAIN.LR = 0.001  #CNN
# config.TRAIN.LR = 0.0001  #simple
config.TRAIN.OPTIMIZER = 'rms'
# config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 20
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.SHUFFLE = False

config.DATASET = edict()
config.DATASET.ROOT = '../dhp2'
config.DATASET.DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False
config.DATASET.FLIP = False
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

config.MODEL = edict()
config.MODEL.STYLE = 'pytorch'
config.MODEL.NUM_JOINTS = 13
config.MODEL.IMAGE_SIZE = [192, 256]  #simple
# config.MODEL.IMAGE_SIZE = [260, 344]  #cnn
config.MODEL.EXTRA = edict()
config.MODEL.EXTRA.NUM_LAYERS = 18
config.MODEL.EXTRA.NUM_DECONV_FILTERS = [256, 256, 256]
config.MODEL.EXTRA.NUM_DECONV_KERNELS = [4, 4, 4]
config.MODEL.EXTRA.HEATMAP_SIZE = [64, 64] #simple
# config.MODEL.EXTRA.HEATMAP_SIZE = [260, 344]  #cnn
config.MODEL.EXTRA.SIGMA = 2
config.MODEL.EXTRA.DECONV_WITH_BIAS = False
config.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL.EXTRA.TARGET_TYPE = 'gaussian'
config.MODEL.EXTRA.NUM_DECONV_LAYERS = 3
config.MODEL.PRETRAINED = 'pose_resnet_50_256x256.pth.tar'
config.MODEL.INIT_WEIGHTS = False



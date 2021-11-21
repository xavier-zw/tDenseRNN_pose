import torch
from torchvision import transforms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
class CFG:
    MODEL_NAME = "tDense"
    SAVE_PATH = MODEL_NAME + "_test"
    result_txt = SAVE_PATH + ".txt"
    ROOT_TRAIN = '/home/datasets/ZJUT/Verified_train'
    ROOT_VALID = '/home/datasets/ZJUT/Verified_valid'
    OUTPUT_DIR = ''
    save_show_image = "show"


    device_ids = [0, 1, 2, 3]
    temporal = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WORKERS = 0

    PRINT_FREQ = 100
    TRAIN = True
    INIT_WEIGHTS = True
    #LR = 0.00001# pose_resnet
    #LR = 0.00001  #sim_lstm
    LR = 0.00005
    #simple_baseline
    OPTIMIZER = 'adam'
    MOMENTUM = 0.9
    WD = 0.0001
    NESTEROV = False
    LR_FACTOR = 0.1
    Hidden_dim = 256
    BEGIN_EPOCH = 0
    END_EPOCH = 1000
    BATCH_SIZE = 64
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    SCALE_FACTOR = 0.25
    ROT_FACTOR = 30

    USE_TARGET_WEIGHT = True

    STYLE = 'pytorch'
    NUM_JOINTS = 13
    IMAGE_SIZE = [256, 256]  # simple
    SOURCE_SIZE = [1280, 800]
    # IMAGE_SIZE = [260, 344]  #cnn

    NUM_LAYERS = 18
    NUM_DECONV_FILTERS = [256, 256, 256]
    NUM_DECONV_KERNELS = [4, 4, 4]
    HEATMAP_SIZE = [64, 64]  # simple
    # HEATMAP_SIZE = [260, 344]  #cnn
    SIGMA = 2
    DECONV_WITH_BIAS = False
    FINAL_CONV_KERNEL = 1
    TARGET_TYPE = 'gaussian'
    NUM_DECONV_LAYERS = 3
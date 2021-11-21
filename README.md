## Environment
The code is developed using python 3.8 on Ubuntu 20.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA 2080ti GPU cards. 

## Quick start
### Installation
1. Install pytorch >= v1.6.0 following [official instruction](https://pytorch.org/).  
   - **Tested with pytorch v1.6.0**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Init output(training model output directory):

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── dataset
   ├── dhp19
   ├── models
   ├── tools
   ├── show
   ├── output
   ├── valid_model 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/open?id=1bdXVmYrSynPLSk5lptvgyQ8fhziobD50) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW4AwKRMklXVzndJT0))
   ```
   ${POSE_ROOT}
    `-- valid_model
        tDense.pth
   ```
   
### Data preparation

**For test data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on COCO val2017 dataset using model zoo's models ([GoogleDrive](https://drive.google.com/drive/folders/1X9-TzWpwbX2zQf2To8lB-ZQHMYviYYh6?usp=sharing))
 

For single-scale testing:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth
```

By default, we use horizontal flip. To test without flip:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.FLIP_TEST False
```

Multi-scale testing is also supported, although we do not report results in our paper:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.SCALE_FACTOR '[0.5, 1.0, 2.0]'
```


#### Training on COCO train2017 dataset

```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

By default, it will use all available GPUs on the machine for training. To specify GPUs, use

```
CUDA_VISIBLE_DEVICES=0,1 python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 

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

7. Download pretrained models from our model tDense.pth and valid data([GoogleDrive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=sharing).
   ```
   ${POSE_ROOT}
    `-- valid_model
        tDense.pth
    `-- valid_data
        *
   ```
   
### Data preparation

**For test data**, please download from ([GoogleDrive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- sample_train
    `-- |-- A*P*/S00
            `-- color
            |   |-- 000001.png
            |   |-- 000002.png
            |   |-- 000003.png
            |   |-- ... 
            `-- depth_raw
            `-- event
            `-- image_event_binary
            `-- label_color_fill
            `-- label_event_fill
            
```

### Training and Testing

#### Testing on CDEHP valid_data dataset using model ([GoogleDrive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=sharing))
 

```
python valid_action.py \
    --modelName tDense \
    --modelDir ./tDense_pose/valid_model/tDense.pth \
    --dataDir ./tDense_pose/valid_data
```

#### Training on COCO train2017 dataset

```
python valid_action.py \
    --modelName tDense \
    --rootTrain ./data/sample_train \
    --rootValid ./data/sample_valid \
    --Temporal 4
```

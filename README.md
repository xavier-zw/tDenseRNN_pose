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
   ├── data
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
    `-- data
        sample_valid
        sample_train
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

#### Testing on CDEHP sample_valid dataset using model and data ([GoogleDrive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=sharing))
 

```
python pose_train.py \
    --modelName tDense \
    --modelDir ./valid_model/tDense.pth \
    --dataDir ./data/sample_valid
```

#### Training on CDEHP sample_train dataset using data ([GoogleDrive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=sharing))

```
python pose_valid.py \
    --modelName tDense \
    --rootTrain ./data/sample_train \
    --rootValid ./data/sample_valid \
    --Temporal 4
```

#### Visualisation of some of the test data.

```
python valid_action.py \
    --modelName tDense \
    --modelDir ./valid_model/tDense.pth \
    --dataDir ./data/sample_valid/A0017P0005/S00
```
The output is saved in the show folder.You can convert the output to video and gif files using image_video.py, which works as follows:
<p align='center'>
	<img src="./show.gif" style="zoom:100%;" />
</p>

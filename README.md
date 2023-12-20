# A Temporal Densely Connected Recurrent Network for Event-based Human Pose Estimation

Event camera is an emerging bio-inspired vision sensors that report per-pixel brightness changes asynchronously. It holds noticeable advantage of high dynamic range, high speed response, and low power budget that enable it to best capture local motions in uncontrolled environments. This motivates us to unlock the potential of event cameras for human pose estimation, as the human pose estimation with event cameras is rarely explored. Due to the novel paradigm shift from conventional frame-based cameras, however, event signals in a time interval contain very limited information, as event cameras can only capture the moving body parts and ignores those static body parts, resulting in some parts to be incomplete or even disappeared in the time interval. This paper proposes a novel densely connected recurrent architecture to address the problem of incomplete information. By this recurrent architecture, we can explicitly model not only the sequential but also non-sequential geometric consistency across time steps to accumulate information from previous frames to recover the entire human bodies, achieving a stable and accurate human pose estimation from event data. Moreover, to better evaluate our model, we collect a large-scale multimodal event-based dataset that comes with human pose annotations, which is by far the most challenging one to the best of our knowledge. The experimental results on two public datasets and our own dataset demonstrate the effectiveness and strength of our approach. Code is available online for facilitating the future research.

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
   ├── models
   ├── tools
   ├── show
   ├── valid_model 
   ├── README.md
   └── requirements.txt
   ```

7. The pretrained models from our model tDense.pth and data.
   ```
   ${POSE_ROOT}
    `-- valid_model
        tDense.pth
    `-- data
        sample_valid
        sample_train
   ```
   
### Data preparation

**For train and valid data*.
```
${POSE_ROOT}
|-- data
`-- |-- sample_train
    `-- |-- A*P*/S00
            `-- image_event_binary
            |   |-- 000001.png
            |   |-- 000002.png
            |   |-- 000003.png
            |   |-- ... 
            `-- label_event_fill
            
```

### Training and Testing
Our pre-trained model and some sample data are available for download on [Google Drive](https://drive.google.com/drive/folders/1rfaQ4h2xJx8wlbnXTl5-VCzE5iQpCtN9?usp=drive_link). 
#### Testing on CDEHP sample_valid dataset using pretrained model from our model tDense.pth.
 

```
python pose_valid.py \
    --modelName tDense \
    --modelDir ./valid_model/tDense.pth \
    --dataDir ./data/sample_valid
    --batch_size 4
```

#### Training on CDEHP sample_train dataset.

```
python pose_train.py \
    --modelName tDense \
    --rootTrain ./data/sample_train \
    --rootValid ./data/sample_valid \
    --Temporal 4
    --batch_size 64
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

 ## Citation
 If you find this code useful, please cite our work with the following bibtex:
 ```
 @article{Shao2023pr,
title = {A Temporal Densely Connected Recurrent Network for Event-based Human Pose Estimation},
journal = {Pattern Recognition},
volume = {##},
pages = {##},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.110048},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323007458?via%3Dihub},
author = {Zhanpeng Shao, Xueping Wang, Wen Zhou, Wuzhen Wang, Jianyu Yang, Youfu Li},
 ```

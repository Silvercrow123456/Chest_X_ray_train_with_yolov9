# Chest_X_ray_train_with_yolov9
Powered by google colab L4 GPU
In the repo, we are going to train yolov9 to make diagnosis on a plain film CXR
First, you can download ChestX‐Det10, a subset of NIH ChestX‐14 from [this website](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset).
Then, we have to convert the .json file to .xlsx file through [this website](https://data.page/json/csv) so that we can modify the content.
```
!python 
```
After converting the train.xlsx to .csv through [this website](https://cloudconvert.com/xlsx-converter), we can upload the images and labels to our google cloud. I created a folder called training_data
## Data pre-processing
First, we have to mount our drive to the colab
```
from google.colab import drive
drive.mount('/content/gdrive')
```
Import helpful packages
```
import cv2
from PIL import Image
import numpy as np
import os
import random
import shutil
from glob import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
test_dir = '/content/gdrive/MyDrive/training_data/test'
train_dir = '/content/gdrive/MyDrive/training_data/train'
train_df = pd.read_csv('/content/gdrive/MyDrive/training_data/train.csv')
train_df.head(2)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/form1.png)
## Training Custom Model
### Cloning Yolo V9 From Github
```
HOME = "/content"  ## Get The Current Working Directory
print(HOME)
os.chdir(HOME)

!git clone https://github.com/WongKinYiu/yolov9.git
%cd yolov9
!pip install -r requirements.txt
```
### Downloading Model Weights
```
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt
!wget -P {HOME}/weights -q https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt
```

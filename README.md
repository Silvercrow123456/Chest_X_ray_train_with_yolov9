# Chest_X_ray_train_with_yolov9
Powered by google colab L4 GPU
In the repo, we are going to train yolov9 to make diagnosis on a plain film CXR
First, you can download ChestX‐Det10, a subset of NIH ChestX‐14 from [this website](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset). \
Then, we have to convert the .json file to .xlsx file through [this website](https://data.page/json/csv) so that we can modify the content.
```
!python 
```
After converting the train.xlsx to .csv, we can upload the file to our c
First, we have to mount our drive to the colab
```
from google.colab import drive
drive.mount('/content/gdrive')
```
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
test_dir = '/content/gdrive/MyDrive/training_data/test'
train_dir = '/content/gdrive/MyDrive/training_data/train'
train_df = pd.read_csv('/content/gdrive/MyDrive/training_data/train.csv')
train_df.head(2)
```
### Training Custom Model

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
Import helpful packages and link google drive to our virtual machine
```
import cv2
from PIL import Image
import numpy as np
import os
import random
import shutil
from glob import glob
import glob
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
mapping image_id to its image_path
```
# Enable pandas progress_apply
tqdm.pandas()

# Load the list of training DICOM images
yy = glob.glob(train_dir + "/*")

# Apply progress_apply to create the 'ImagePath' column
train_df['ImagePath'] = train_df['image_id'].progress_apply(lambda x: next(filter(lambda y: x in y, yy), None))

# Filter out the 'No Finding' class (class_id == 14)
train_df = train_df[train_df['class_name'] != 'No finding'].reset_index(drop=True)

# Select only required columns
train_df = train_df[['ImagePath', 'image_id', 'class_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max']]
```
Visualize training dataframe, this block is elective
```
print("No Of The Unique ImagePath :--->", len(set(train_df['ImagePath'])))
print("Shape Of The Data Frame :->", train_df.shape)
train_df.head(2)
```
！[image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/form2.png)
```
png_paths =  list(set(train_df['ImagePath'])) #-> this move is important

#Turn pngs to numpy array so we can better transfer then to our virtual machine data folder
from numpy import asarray
def png2array(path):

    img = Image.open(path)
    numpydata = asarray(img)
    return numpydata

#This function is elective
def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", cmap='gray', img_size=(500,500)):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

imgs = [png2array(path) for path in png_paths[:4]]
plot_imgs(imgs)
```

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

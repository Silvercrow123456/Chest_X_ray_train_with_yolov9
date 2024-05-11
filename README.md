# Chest_X_ray_train_with_yolov9
Powered by google colab L4 GPU
In the repo, we are going to train yolov9 to make diagnosis on a plain film CXR
First, you can download ChestX‐Det10, a subset of NIH ChestX‐14 from [this website](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset).
Then, we have to convert the .json file to .xlsx file through [this website](https://data.page/json/csv) so that we can modify the content.
```
!python 
```
After converting the train.xlsx to .csv through [this website](https://cloudconvert.com/xlsx-converter), we can upload the images and labels to our google cloud. I created a folder called training_data
## Loading needed images(with abnormal findings) into virtual machine data folder
### Mount our drive to the colab
```
from google.colab import drive
drive.mount('/content/gdrive')
```
### Import helpful packages and create links to our virtual machine
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
### Mapping image_id to its image_path
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
Visualize training dataframe
```
# this block is elective
print("No Of The Unique ImagePath :--->", len(set(train_df['ImagePath'])))
print("Shape Of The Data Frame :->", train_df.shape)
train_df.head(2)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/form2.png)
### Function to save the images
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
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/CXRs.png)
Slower way
```
from skimage import exposure

def saving_image(output_dir, png_path_list):
    for png_path in tqdm(png_path_list):
        file_name = os.path.basename(png_path).split('.')[0]
        image_array = png2array(png_path)
        cv2.imwrite(os.path.join(output_dir, f"{file_name}.png"), image_array)
```
Use MultiThreading to Process and save the image in a faster way
```
Not yet upload
```
### Create data folder CXR and load traing images into subfolder "train"
```
output_dir = "/content/CXR/train"
os.makedirs(output_dir, exist_ok=True)

# Call the function to save images
saving_image(output_dir, png_paths)
```
### Load testing images into subfolder "test"
```
test_png_paths = glob.glob("/content/gdrive/MyDrive/training_data/test/*")

output_dir = "/content/CXR/test"
os.makedirs(output_dir, exist_ok=True)

# Call the function to save images
saving_image(output_dir, test_png_paths)
```
Check if the images is successfully loading into the content/CXR/train folder
```
yy = glob.glob("/content/CXR/train/*") ## SaveImagePath
array = cv2.imread(yy[2])
plt.imshow(array)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/CXRs_in_folder.png)
## Generate coco style labels with normalized bounding boxes
### Get image sizes
```
heights = []
widths =  []

# Use list comprehensions for a more concise and efficient code
heights = [cv2.imread(i).shape[0] for i in tqdm(yy)]
widths = [cv2.imread(i).shape[1] for i in tqdm(yy)]

print("Height is ", heights[:3])
print("width is ", widths[:3])
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/height_width.png)
### Creating Dataframe which will contain the columns SaveImagePath , heights and width
```
# Now Creating Data Frame For The Height ,Width
df = pd.DataFrame(yy, columns =['SaveImagePath'])
df['image_id'] = df['SaveImagePath'].apply(lambda x: x.split('/')[-1].split('.')[0]) +'.png' # WT shit???
df['Height'] = heights
df['Width']  = widths

print("shape Of The Data Frame :->", df.shape)
print("shape of the train_df", train_df.shape)

# Merge the 'df' data frame and merge it with the original one 
final_df  = train_df.merge(df, on = 'image_id')
final_df.head(2)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/merged_data_form.png)
### This Function Will Return The DataFrame With Normalize Bounding Box
```
def convert_to_yolo_format(df):
    # Normalize the bounding box coordinates
    df['center_x'] = (df['x_min'] + df['x_max']) / 2
    df['center_y'] = (df['y_min'] + df['y_max']) / 2
    df['b_box_width'] = df['x_max'] - df['x_min']
    df['b_box_height'] = df['y_max'] - df['y_min']

    # Calculate normalized coordinates and dimensions
    df['normalized_x'] = df['center_x'] / df['Width']
    df['normalized_y'] = df['center_y'] / df['Height']
    df['normalized_width'] = df['b_box_width'] / df['Width']
    df['normalized_height'] = df['b_box_height'] / df['Height']

    return df

df_yolo = convert_to_yolo_format(final_df)
df_yolo.head(2)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/df_yolo_with_normalize.png)
### Create subfolder to store the txt labels
```
Image_label_dir = "/content/CXR/labels"
os.makedirs(Image_label_dir, exist_ok = True)

print("Storing  Training Image BoundingBox","-"*50)
print("Train label dir is ", Image_label_dir)

#Function to save bounding boxes
for file in tqdm(yy):
    filename = file.split('/')[-1].split('.')[0]
    output_file = Image_label_dir+"/"+filename+".txt"
    # Open the output file for writing
    with open(output_file, 'w') as f:
        # Iterate over the filtered DataFrame and write bounding box information to the file
        for _, row in df_yolo.iterrows():
            if (row['image_id'].split('.')[0]) == filename:
                class_id = int(row['class_id'])
                x_center, y_center = row['normalized_x'], row['normalized_y']
                width, height = row["normalized_width"], row['normalized_height']
                f.write(f"{class_id}\t{x_center}\t{y_center}\t{width}\t{height}\n")
```
### Finally We Have store Images and Labels in the CXR folder~ 
```
#you can use the following code to store the labels into our google cloud so you don't have to re-create labels again next time
!cp -r '/content/CXR/labels' '/content/gdrive/MyDrive/training_data'
```
make path to labels folder
```
#This move is important since you can check wether the .txt label is correctly generated
label_files = glob.glob('/content/CXR/labels/*')

with open(label_files[0]) as f:
    file = f.read()
    print(file)
```
### Combine the class_id and class_name(disease name)
```
dict_ = dict(zip(df_yolo['class_id'], df_yolo['class_name']))
print(dict_)
```
### Plot bounding box on CXRs
Now if you run the following code, you will found that images and labels are stored in a different order
```
label_files[0], yy[0] 
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/different_order.png)
So we have to use functions to re-order the image file and matches to label files
```
print("first 3 label_files :->", label_files[:3])

image_files = []
for label in label_files:
    label_name = label.split('/')[-1].split('.')[0]
    image_files.extend([i for i in yy if label_name in i])

print("-"*90)
print("first 3 image_files", image_files[:3])
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/reshape1.png)
shape it on df data frame
```
df = pd.DataFrame(list(zip(label_files, image_files)), columns=['label_files', 'image_files'])
print(df.shape)
df.head(2)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/reshape2.png)
### Now We have corrected the label of the image files and label files
```
df.label_files[0], df.image_files[0]
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/reshape3.png)
### Plot bounding box on the image
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_image_with_bounding_box(image, bounding_boxes, class_dict):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for box in bounding_boxes:
        class_id, x, y, width, height = map(float, box.split())
        image_width, image_height = image.shape[1], image.shape[0]
        x1 = int((x - width / 2) * image_width)
        y1 = int((y - height / 2) * image_height)
        x2 = int((x + width / 2) * image_width)
        y2 = int((y + height / 2) * image_height)

        # Choose random color for bounding box
        color = [random.random() for _ in range(3)]

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add label text using class label from dictionary
        label_text = class_dict[int(class_id)]
        ax.text(x1, y1, label_text, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

    plt.show()

def main(image_path, bounding_box_path, class_dict):
    # Read image
    image = cv2.imread(image_path)

    # Read bounding boxes
    with open(bounding_box_path, 'r') as file:
        bounding_boxes = file.readlines()

    # Plot image with bounding boxes
    plot_image_with_bounding_box(image, bounding_boxes, class_dict)

#img_no can change as want
img_no = 44
main(df.image_files[img_no], df.label_files[img_no], class_dict = dict_)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/reshape4.png)
### This df is the actual train_data. From this we will create train(2301 images) and valid(19 image)
the ratio of train and valid can be customerized
```
from sklearn.model_selection import train_test_split
df_train, df_valid = train_test_split(df, test_size = 19 , random_state = 42)
print("shape of df_train", df_train.shape)
print("shape of df_valid", df_valid.shape)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/split.png)
check if valid list was successfully maked
```
df_valid.head(1)
```
![image](https://github.com/Silvercrow123456/Chest_X_ray_train_with_yolov9/blob/main/Illustrations/valid1.png)
### Create "dataset" folder and its subset: train (image/labels) and valid (image/labels) and shift the images and labels from "CXR" folder into it
```
import os
import shutil

# Creating directories for train and validation images and labels
os.makedirs("/content/dataset/train/images", exist_ok=True)
os.makedirs("/content/dataset/valid/images", exist_ok=True)
os.makedirs("/content/dataset/train/labels", exist_ok=True)
os.makedirs("/content/dataset/valid/labels", exist_ok=True)

# Copying train images
print("COPYING TRAIN IMAGES :-->", "-"*50)
for img_path in df_train['image_files']:
    shutil.move(img_path, "/content/dataset/train/images")

# Copying validation images
print("COPYING VALID IMAGES :-->", "-"*50)
for img_path in df_valid['image_files']:
    shutil.move(img_path, "/content/dataset/valid/images")

# Copying train labels
print("COPYING TRAIN LABELS :-->", "-"*50)
for label_path in df_train['label_files']:
    shutil.move(label_path, "/content/dataset/train/labels")

# Copying validation labels
print("COPYING VALID LABELS :-->", "-"*50)
for label_path in df_valid['label_files']:
    shutil.move(label_path, "/content/dataset/valid/labels")
```
## Image augmentation
We use this step to increase the amount of training data using data augmentation in order to imporve the performance. First, we have to download the imgaug.py from repo and up load to colab virtual machine
```
os.makedirs("/content/dataset/train_aug/images", exist_ok=True)
os.makedirs("/content/dataset/train_aug/labels", exist_ok=True)
!python imgaug.py --input_img '/content/dataset/train/images/' --output_img '/content/dataset/train_aug/images/' --input_label '/content/dataset/train/labels/' --output_label '/content/dataset/train_aug/labels/'
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

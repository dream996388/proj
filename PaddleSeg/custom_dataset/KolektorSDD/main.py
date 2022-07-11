import os
import shutil

import cv2
import numpy as np
from anno import *

label_img=None
ori_img=None
dataset_split='val'
# list dir names in the directory
path=rf'D:\AI\colab\proj\PaddleSeg\custom_dataset\KolektorSDD\\{dataset_split}'
dir_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
# print(len(dir_names))
text=''

for dir_name in dir_names:
    dir_path = os.path.join(path, dir_name)
    # if file extension is '.bmp', read this image as label image
    for file in os.listdir(dir_path):
        if file.endswith(".bmp"):
            label_img = cv2.imread(os.path.join(dir_path, file))
            colormap=label_colormap(255)
            label_img_pil=to_indeximg(label_img,colormap)
            cv2.imwrite(os.path.join(dir_path, file.replace('.bmp', '.png')), label_img)
            label_img_pil.save(os.path.join(dir_path, file.replace('.bmp', '.png')))
            # delete the file
            os.remove(os.path.join(dir_path, file))
    # create a blank image with the size '500x500'
    # ori_img = np.zeros((500, 500, 3), np.uint8)
    # show ori_img
    # cv2.imshow('ori_img', ori_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




# black_list=[]
# for dir_name in dir_names:
#     # list file names in the directory, if file extension is '.jpg'
#     dir_path = os.path.join(path, dir_name)
#     for file in os.listdir(dir_path):
#         if file.endswith(".jpg"):
#             label_filename=file.split('.')[0]+'_label.png'
#             text= text + f'{dataset_split}/{dir_name}/{file} {dataset_split}/{dir_name}/{label_filename}\n'
# print(text)
#
#
#
# # write the text to a file, named as 'train_list.txt'
# with open(f'{dataset_split}_list.txt', 'w') as f:
#     f.write(text)

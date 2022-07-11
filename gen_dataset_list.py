import os
import shutil
from configs import *

src_path=r'D:\AI\v1_dataset'
text=''
dataset='train'
dir_names = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]
# print(dir_names)
for kos_dir_name in dir_names:
    part_dir_names = [f for f in os.listdir(os.path.join(src_path, kos_dir_name)) if os.path.isdir(os.path.join(src_path, kos_dir_name, f))]
    # print(part_dir_names)
    for part_dir_name in part_dir_names:
        text = text + f'{dataset}/{kos_dir_name}/{part_dir_name}/output.jpg {dataset}/{kos_dir_name}/{part_dir_name}/label.png\n'

print(dataset)


# write the text to a file, named as 'train_list.txt'
with open(f'{dataset}_list.txt', 'w') as f:
    f.write(text)

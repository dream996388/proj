import os
import time
from cv_utils import *
from configs import *
import cv2 as cv


src_path = r'D:\AI\real image\KolektorSDD'
dst_dir = fr'D:\AI\colab\proj\output'
dst_path = dst_dir
sf_py_path = r'D:\AI\colab\proj\Texture-Reformer\transfer.py'

defect_img=None
defect_roi=None
label_img=None
bk_img=None

def main():
    for i in range(len(train_defect_list)):
        dir_name = train_defect_list[i].split('-')[0]
        img_name = train_defect_list[i].split('-')[1]
        defect_img_path = os.path.join(src_path, dir_name, img_name + '.jpg')
        label_img_path = os.path.join(src_path, dir_name, img_name + '_label.bmp')
        defect_img = cv.imread(defect_img_path, cv.IMREAD_GRAYSCALE)
        label_img = cv.imread(label_img_path, cv.IMREAD_GRAYSCALE)
        ori_defect, label_defect, _ = extract_roi_from_img(defect_img, label_img, top_npixel=20,bottom_npixel=250)
        cv.imwrite(r'D:\AI\colab\Texture Reformer\main.py\inputs\sdd\part29\style-transfer4\style.jpg',ori_defect[0])
        cv.imwrite(r'D:\AI\colab\Texture Reformer\main.py\inputs\sdd\part29\style-transfer4\style-seg.png',label_defect[0])

if __name__ == '__main__':
    st = time.time()
    main()
    end = time.time()
    print('spent time:', end - st)
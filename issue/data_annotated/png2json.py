# requirements:
# scikit-image
import shutil

from labelme import shape
import os
import subprocess
from labelme.utils import img_arr_to_b64
import cv2 as cv
from skimage import measure
import numpy as np
import json
import base64
from labelme import LabelFile

### ******************* ####
# import base64
# from labelme import LabelFile
# img_path='data_annotated/output.jpg'
# imgBytes = LabelFile.load_image_file(img_path)
# print(base64.b64encode(imgBytes).decode('utf-8'))
### ******************* ####
from issue.data_annotated.label_rgb import label_colormap
from label_rgb import to_indeximg

img = None
mask = None
data = None
img_path = ''
label = ''
out_filename = 'output_json/img.json'
points = []
height = -1
width = -1

src_dir = 'D:\AI\output'
dst_dir = 'D:\AI\output'
labelme_path = r'D:\AI\colab\proj\annotation\labelme.exe'
json2dataset_path = r'D:\AI\colab\proj\annotation\labelme_json_to_dataset.exe'


def get_points(mask):
    points = []
    _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    # cv.imshow('mask',mask)
    # cv.waitKey(0)
    # contours = measure.find_contours(mask, 0.25)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=len, reverse=True)[:1]
    for n, contour in enumerate(contours):
        coords = contour
        # coords = measure.approximate_polygon(contour, tolerance=0.5)[:-1]
        segmentation = np.flip(coords, axis=1).tolist()
    for seg in segmentation:
        points.append(seg[0])

    return points


def get_data(img_path):
    imgBytes = LabelFile.load_image_file(img_path)
    return base64.b64encode(imgBytes).decode('utf-8')


def pngtojson(img_path, mask_path, label, out_filename):
    img_path = img_path
    mask_path = mask_path
    img = cv.imread(img_path)
    mask = cv.imread(mask_path, 0)
    points = get_points(mask)
    data = get_data(img_path)
    label = label
    height = img.shape[0]
    width = img.shape[1]
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": points,
                "group_id": 'null',
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": img_path,
        "imageData": data,
        "imageHeight": height,
        "imageWidth": width
    }

    with open(out_filename, 'w') as outfile:
        json.dump(annotation, outfile, indent=2)

    # print(annotation)


def post_processing(lable_path):
    colormap = label_colormap(255)
    label_img = cv.imread(lable_path, cv.IMREAD_GRAYSCALE)
    _, label_img = cv.threshold(label_img, 0, 1, cv.THRESH_BINARY)
    contours, _ = cv.findContours(label_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(label_img, contours, -1, 1, -1)
    label_img_pil = to_indeximg(label_img, colormap)

    return label_img_pil


done_edit = ['D:\\AI\\output\\kos01\\Part0', 'D:\\AI\\output\\kos01\\Part0_1', 'D:\\AI\\output\\kos01\\Part2',
             'D:\\AI\\output\\kos01\\Part2_1', 'D:\\AI\\output\\kos01\\Part2_2', 'D:\\AI\\output\\kos01\\Part4',
             'D:\\AI\\output\\kos01\\Part7', 'D:\\AI\\output\\kos01\\Part7_1', 'D:\\AI\\output\\kos01\\Part7_2',
             'D:\\AI\\output\\kos01\\Part7_3', 'D:\\AI\\output\\kos02\\Part0', 'D:\\AI\\output\\kos02\\Part2',
             'D:\\AI\\output\\kos02\\Part3', 'D:\\AI\\output\\kos02\\Part3_1', 'D:\\AI\\output\\kos03\\Part0',
             'D:\\AI\\output\\kos03\\Part1', 'D:\\AI\\output\\kos03\\Part3', 'D:\\AI\\output\\kos03\\Part3_1',
             'D:\\AI\\output\\kos03\\Part3_2', 'D:\\AI\\output\\kos03\\Part4', 'D:\\AI\\output\\kos03\\Part5',
             'D:\\AI\\output\\kos03\\Part6', 'D:\\AI\\output\\kos03\\Part7', 'D:\\AI\\output\\kos04\\Part0',
             'D:\\AI\\output\\kos04\\Part0_1', 'D:\\AI\\output\\kos04\\Part1', 'D:\\AI\\output\\kos04\\Part1_2',
             'D:\\AI\\output\\kos04\\Part6', 'D:\\AI\\output\\kos04\\Part6_1', 'D:\\AI\\output\\kos04\\Part6_2',
             'D:\\AI\\output\\kos05\\Part0', 'D:\\AI\\output\\kos05\\Part1_1', 'D:\\AI\\output\\kos05\\Part3',
             'D:\\AI\\output\\kos05\\Part4_1', 'D:\\AI\\output\\kos05\\Part7', 'D:\\AI\\output\\kos05\\Part7_1',
             'D:\\AI\\output\\kos06\\Part0', 'D:\\AI\\output\\kos06\\Part0_2', 'D:\\AI\\output\\kos06\\Part2',
             'D:\\AI\\output\\kos06\\Part2_1', 'D:\\AI\\output\\kos06\\Part3', 'D:\\AI\\output\\kos06\\Part3_1',
             'D:\\AI\\output\\kos06\\Part5', 'D:\\AI\\output\\kos07\\Part0', 'D:\\AI\\output\\kos07\\Part5',
             'D:\\AI\\output\\kos07\\Part6', 'D:\\AI\\output\\kos07\\Part6_1', 'D:\\AI\\output\\kos08\\Part5',
             'D:\\AI\\output\\kos08\\Part5_1', 'D:\\AI\\output\\kos08\\Part6', 'D:\\AI\\output\\kos08\\Part6_1',
             'D:\\AI\\output\\kos08\\Part7']
blacklist = ['D:\\AI\\output\\kos02\\Part3_2', 'D:\AI\output\kos04\Part1_1', 'D:\AI\output\kos04\Part2',
             'D:\AI\output\kos04\Part4', 'D:\AI\output\kos04\Part7', 'D:\AI\output\kos04\Part7_1',
             'D:\AI\output\kos05\Part1', 'D:\AI\output\kos05\Part4', 'D:\AI\output\kos06\Part0_1',
             'D:\AI\output\kos06\Part1', 'D:\AI\output\kos06\Part5_1', 'D:\AI\output\kos06\Part6',
             'D:\AI\output\kos06\Part6_1', 'D:\AI\output\kos06\Part6_2', 'D:\AI\output\kos07\Part0_1',
             'D:\AI\output\kos07\Part4', 'D:\AI\output\kos07\Part6_2', 'D:\AI\output\kos08\Part0',
             'D:\AI\output\kos08\Part4', 'D:\AI\output\kos08\Part6_2']


def main():
    # list all dir in the 'src_dir'
    kos_dirs = os.listdir(src_dir)
    for kos_dir in kos_dirs:
        # list all dir in the 'dir', named as 'kos'
        part_dirs = os.listdir(os.path.join(src_dir, kos_dir))
        for part_dir in part_dirs:
            full_path = os.path.join(src_dir, kos_dir, part_dir)
            print(full_path)
            if full_path in done_edit:
                continue
            elif full_path in blacklist:
                # delete the dir
                shutil.rmtree(full_path)
                print('delete: ' + full_path)
                continue
            else:

                pngtojson(os.path.join(src_dir, kos_dir, part_dir, 'output.jpg'),
                          os.path.join(src_dir, kos_dir, part_dir, 'output_label.png'),
                          'defect',
                          os.path.join(src_dir, kos_dir, part_dir, 'output.json'))
                # open external program
                import subprocess
                output = subprocess.check_output([f'{labelme_path}', f'{os.path.join(src_dir, kos_dir, part_dir)}'])
                output = subprocess.check_output(
                    [f'{json2dataset_path}', '--out', f'{os.path.join(dst_dir, kos_dir, part_dir)}',
                     f'{os.path.join(src_dir, kos_dir, part_dir, "output.json")}'])
                # read 'label.png' image, and process it
                label_img_pil = post_processing(os.path.join(src_dir, kos_dir, part_dir, 'label.png'))
                label_img_pil.save(os.path.join(dst_dir, kos_dir, part_dir, 'label.png'))
                label_img = cv.imread(os.path.join(src_dir, kos_dir, part_dir, 'label.png'))

                # cv.imwrite(os.path.join(dst_dir, kos_dir, part_dir, 'test_output_label.png'), label_img)
                cv.namedWindow("label.png", cv.WINDOW_NORMAL)
                cv.imshow('label.png', label_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                done_edit.append(full_path)
            print(len(done_edit))
            print(done_edit)


if __name__ == '__main__':
    main()

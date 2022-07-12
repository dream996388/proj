import os
import shutil

import cv2 as cv


# D:\\AI\\output\\kos01\\Part0', 'D:\\AI\\output\\kos01\\Part0_1', 'D:\\AI\\output\\kos01\\Part2',
#              'D:\\AI\\output\\kos01\\Part2_1', 'D:\\AI\\output\\kos01\\Part2_2', 'D:\\AI\\output\\kos01\\Part4',
#              'D:\\AI\\output\\kos01\\Part7', 'D:\\AI\\output\\kos01\\Part7_1', 'D:\\AI\\output\\kos01\\Part7_2',
# 'D:\\AI\\output\\kos01\\Part7_3'
# 'D:\\AI\\output\\kos02\\Part3'
# D:\AI\colab\proj\v15\syn-kos03\Part3
src_path=r''
dst_path=r'D:\AI\colab\proj\v15'
labelme_path = r'D:\AI\colab\proj\annotation\labelme.exe'
json2dataset_path = r'D:\AI\colab\proj\annotation\labelme_json_to_dataset.exe'

done_edit = ['D:\\AI\\output\\kos02\\Part0', 'D:\\AI\\output\\kos02\\Part2',
             'D:\\AI\\output\\kos02\\Part3_1', 'D:\\AI\\output\\kos03\\Part0',
             'D:\\AI\\output\\kos03\\Part1' , 'D:\\AI\\output\\kos03\\Part3_1',
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


def main():
    for img_dir in done_edit:
        # make a new dir, named as 'syn-{}'
        kos_dir=img_dir.split('\\')[-2]
        part_dir=img_dir.split('\\')[-1]
        # copy 'img_dir' to dst path
        dst_dir=os.path.join(dst_path, f'syn-{kos_dir}',part_dir)
        if os.path.exists(dst_dir):
            continue
        else:
            shutil.copytree(img_dir, dst_dir)
        # open external program
        print(dst_dir)
        import subprocess
        output = subprocess.check_output([f'{labelme_path}', f'{os.path.join(dst_dir, "output.jpg")}'])
        output = subprocess.check_output(
            [f'{json2dataset_path}', '--out', f'{os.path.join(dst_dir)}',
             f'{os.path.join(dst_dir, "output.json")}'])

        label_viz = cv.imread(os.path.join(dst_dir, 'label_viz.png'))
        cv.imshow('label_viz', label_viz)
        cv.waitKey(0)
        cv.destroyAllWindows()



if __name__ == '__main__':
    main()

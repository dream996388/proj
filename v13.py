import os
import subprocess
import time
from random import randint

import cv2 as cv
import numpy as np
from configs import *
from cv_utils import *

text = ''
img_cnt = 1
M = 5
N = 3
npixel = 0
xs_upper = 0
xs_lower = 0
ys_upper = 0
ys_lower = 0
max_bk_img = 10
xshift = randint(xs_lower, xs_upper)
yshift = randint(ys_lower, ys_upper)
defect_img = None
label_img = None
train_label_img = None
bk_img = None
concat_img = None
content_img = None
content_seg = None
ori_content_seg = None
style_img = None
style_seg = None
ori_defect = None
label_defect = None
ori_biggerbbox_defect = None
label_biggerbbox_defect = None
ori_defect_trans = None
label_defect_trans = None
bboxs = None
extend_defect_img = None
src_path= r'/content/style-transfer/KolektorSDD'
dst_dir= fr'/content/style-transfer/output'
dst_path=dst_dir
sf_py_path=r'/content/style-transfer/Texture-Reformer/transfer.py'

# src_path = r'D:\AI\real image\KolektorSDD'
# dst_dir = fr'D:\AI\colab\proj\output'
# dst_path = dst_dir
# sf_py_path = r'D:\AI\colab\proj\Texture-Reformer\transfer.py'


def main():
    for i in range(len(train_defect_list)):
        dir_name = train_defect_list[i].split('-')[0]
        img_name = train_defect_list[i].split('-')[1]
        defect_img_path = os.path.join(src_path, dir_name, img_name + '.jpg')
        label_img_path = os.path.join(src_path, dir_name, img_name + '_label.bmp')
        # print(defect_img_path)
        # print(label_img_path)

        defect_img = cv.imread(defect_img_path, cv.IMREAD_GRAYSCALE)
        # show defect_img
        # cv.imshow('defect_img', defect_img)
        # cv.waitKey(0)
        label_img = cv.imread(label_img_path, cv.IMREAD_GRAYSCALE)
        ori_defect, label_defect, _ = extract_objs_from_img(defect_img, label_img, npixel=0)
        ori_biggerbbox_defect, label_biggerbbox_defect, bboxs = extract_objs_from_img(defect_img, label_img,
                                                                                      npixel=npixel)
        ori_defect_trans = transform(ori_biggerbbox_defect)  ## 使用條件:物件數量=1
        label_defect_trans = transform(label_biggerbbox_defect)  ## 使用條件:物件數量=1
        # cvtColor label_defect_trans image to RGB image
        # label_defect_trans = [cv.cvtColor(img, cv.COLOR_GRAY2RGB) for img in label_defect_trans]
        # show ori_defect_trans image
        # cv.imshow('defect image',ori_defect_trans[0])
        # cv.waitKey(0)

        # read train_background_list
        for cnt in range(max_bk_img):
            randnum = randint(0, len(train_background_list) - 1)
            dir_name = train_background_list[randnum].split('/')[0]
            img_name = train_background_list[randnum].split('/')[1]
            bk_img_path = os.path.join(src_path, dir_name, img_name)
            bk_img = cv.imread(bk_img_path, cv.IMREAD_GRAYSCALE)
            # show bk_img
            # cv.imshow('bk img', bk_img)
            # cv.waitKey(0)
            extend_bk_img = cv.copyMakeBorder(bk_img, 0, 0, 0, ori_defect_trans[0].shape[1], cv.BORDER_REFLECT)
            # ori_extend_defect_img=cv.copyMakeBorder(ori_defect_trans[0], 0, bk_img.shape[0]-ori_defect_trans[0].shape[0], 0, 0, cv.BORDER_CONSTANT, value=0)
            # label_extend_defect_img=cv.copyMakeBorder(label_defect_trans[0], 0, bk_img.shape[0]-label_defect_trans[0].shape[0], 0, 0, cv.BORDER_CONSTANT, value=0)
            # show ori_extend_defect_img
            # cv.imshow('ori_extend_defect_img', ori_extend_defect_img)
            # cv.waitKey(0)
            # concat_img=cv.hconcat([bk_img,ori_extend_defect_img])
            extend_bk_img[0:ori_defect_trans[0].shape[0],
            extend_bk_img.shape[1] - ori_defect_trans[0].shape[1]:extend_bk_img.shape[1]] = ori_defect_trans[0]
            # show extend_bk_img
            # cv.imshow('extend_bk_img',extend_bk_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # create content img size the same as bk img size
            content_img = np.zeros((bk_img.shape[0], bk_img.shape[1]), dtype=np.uint8)
            # fill content_seg with random color
            ori_content_seg = fill_random_color(bk_img, m=M, n=N)
            content_seg = ori_content_seg.copy()
            train_label_img = content_img.copy()
            # paste label_defect_trans to content_seg image
            for z in range(len(label_defect_trans)):
                x, y, w, h = bboxs[z]
                H, W = bk_img.shape[:2]
                ys_lower = -(bk_img.shape[0] // 2)
                ys_upper = +(bk_img.shape[0] // 2)
                xs_lower = -(bk_img.shape[1] // 2)
                xs_upper = +(bk_img.shape[1] // 2)
                xshift = randint(xs_lower, xs_upper)
                yshift = randint(ys_lower, ys_upper)
                # show xshift and yshift
                # print(xshift,yshift)
                y1 = y + yshift - npixel
                y2 = y + yshift + h + npixel
                x1 = x + xshift - npixel
                x2 = x + xshift + w + npixel
                if y1 < 0:
                    y1 = 0
                    y2 = label_defect_trans[z].shape[0]
                if y2 > H:
                    y1 = H - label_defect_trans[z].shape[0]
                    y2 = H
                if x1 < 0:
                    x1 = 0
                    x2 = label_defect_trans[z].shape[1]
                if x2 > W:
                    x1 = W - label_defect_trans[z].shape[1]
                    x2 = W
                if len(label_defect_trans[z].shape) == 2:
                    content_seg[y1:y2, x1:x2] = cv.add(content_seg[y1:y2, x1:x2],
                                                       cv.cvtColor(label_defect_trans[z], cv.COLOR_GRAY2BGR))
                    # todo:修改label_defect_trans
                    # ValueError: could not broadcast input array from shape (46,246) into shape (46,246,1)
                    train_label_img[y1:y2, x1:x2] = cv.add(train_label_img[y1:y2, x1:x2], label_defect_trans[z])
                else:
                    content_seg[y1:y2, x1:x2] = cv.add(content_seg[y1:y2, x1:x2], label_defect_trans[z])
                    train_label_img[y1:y2, x1:x2] = cv.add(train_label_img[y1:y2, x1:x2], label_defect_trans[z])
            # style img copy concat_img, and convet to bgr image
            style_img = cv.cvtColor(extend_bk_img.copy(), cv.COLOR_GRAY2BGR)
            extend_ori_content_seg = cv.copyMakeBorder(ori_content_seg, 0, 0, 0, ori_defect_trans[0].shape[1],
                                                       cv.BORDER_CONSTANT, value=0)
            extend_ori_content_seg[0:ori_defect_trans[0].shape[0],
            extend_bk_img.shape[1] - ori_defect_trans[0].shape[1]:extend_bk_img.shape[1]] = cv.cvtColor(
                label_defect_trans[0], cv.COLOR_GRAY2BGR)
            style_seg = extend_ori_content_seg.copy()
            _, train_label_img = cv.threshold(train_label_img, 254, 1, cv.THRESH_BINARY)

            # show content_seg, style_seg, content_img, style_img
            # cv.imshow('content_seg',content_seg)
            # cv.imshow('style_seg',style_seg)
            # cv.imshow('content_img',content_img)
            # cv.imshow('style_img',style_img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            # print(defect_img_path)
            # save train_label_img, defect_img, bk_img, content_seg, style_seg, content_img, style_img
            # check if 'dir_name' is exist or not
            dst_path = os.path.join(dst_dir, dir_name, img_name.split('.')[0])
            dir_name_cnt = 0
            while os.path.exists(dst_path):
                dir_name_cnt += 1
                dst_path = os.path.join(dst_dir, dir_name, img_name.split('.')[0] + '_' + str(dir_name_cnt))
            os.makedirs(dst_path)
            cv.imwrite(os.path.join(dst_path, 'output_label.png'), train_label_img)
            cv.imwrite(os.path.join(dst_path, 'defect_img.png'), defect_img)
            cv.imwrite(os.path.join(dst_path, 'bk_img.png'), bk_img)
            cv.imwrite(os.path.join(dst_path, 'content_seg.png'), content_seg)
            cv.imwrite(os.path.join(dst_path, 'style_seg.png'), style_seg)
            cv.imwrite(os.path.join(dst_path, 'content_img.png'), content_img)
            cv.imwrite(os.path.join(dst_path, 'style_img.png'), style_img)
            #
            # # 開始生成影像
            # # get external program output
            st = time.time()
            output = subprocess.check_output(['python', sf_py_path,
                                              '-content', os.path.join(dst_path, 'content_img.png'),
                                              '-content_sem', os.path.join(dst_path, 'content_seg.png'),
                                              '-style', os.path.join(dst_path, 'style_img.png'),
                                              '-style_sem', os.path.join(dst_path, 'style_seg.png'),
                                              '-outf', os.path.join(dst_path),
                                              '-enhance_alpha', '0.1',
                                              '-enhance', 'adain',
                                              '-out_filename', 'output.jpg',
                                              '-fine_alpha', '1.5'])
            end = time.time()
            print('spent time:', end - st)
            # print(output)

            # # show content image
            # cv.imshow('result.jpg', )
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            global img_cnt
            print('processed %d images' % img_cnt)
            print('-----------------------------------------------------')
            img_cnt += 1
            if img_cnt % 10 == 0:
                import torch
                torch.cuda.empty_cache()
                print('free gpu memory')
                # print('cuda memory:', torch.cuda.memory_allocated())
                # print('cuda cached:', torch.cuda.memory_cached())
                # print('cuda total:', torch.cuda.get_device_properties(0).total_memory)
                # print('cuda free:', torch.cuda.get_device_properties(0).free_memory)
                # print('cuda used:', torch.cuda.get_device_properties(0).total_memory - torch.cuda.get_device_properties(0).free_memory)
                # print('-----------------------------------------------------')


            #     if len(ndarr.shape) == 3:
            #         ndarr=cv.cvtColor(ndarr, cv.COLOR_BGR2GRAY)
            #         # if 'ndarr' height is even number, extend it
            #         if ndarr.shape[0] % 2 == 0:
            #              ndarr=cv.copyMakeBorder(ndarr, 1, 5, 0, 4, cv.BORDER_REPLICATE)
            #         else:
            #              ndarr=cv.copyMakeBorder(ndarr, 2, 4, 0, 4, cv.BORDER_REPLICATE)
            #         cv.imwrite(filename, ndarr)
            #     else:
            #         cv.imwrite(filename, ndarr)

            # check image size
            img = cv.imread(os.path.join(dst_path, 'output.jpg'), cv.IMREAD_GRAYSCALE)
            if img.shape[0] < train_label_img.shape[0]:
                # print(f'image size not match {img.shape} != {train_label_img.shape}')
                # dsize = (width, height)
                train_label_img = cv.resize(train_label_img, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
            elif img.shape[0] > train_label_img.shape[0]:
                # print(f'image size not match {img.shape} != {train_label_img.shape}')
                # dsize = (width, height)
                img = cv.resize(img, (train_label_img.shape[1], train_label_img.shape[0]), interpolation=cv.INTER_CUBIC)
            # 保存縮放後的影像:output.jpg output_label.png
            cv.imwrite(os.path.join(dst_path, 'output.jpg'), img)
            cv.imwrite(os.path.join(dst_path, 'output_label.png'), train_label_img)
















if __name__ == '__main__':
    st = time.time()
    main()
    end = time.time()
    print('spent time:', end - st)

import cv2 as cv
import numpy as np

# 定义平移translate函数
def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 返回转换后的图像
    return shifted

def extract_objs_from_img(ori_img,label_img,npixel):
    gray=label_img.copy()
    ori_rois=[]
    label_rois=[]
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # get bounding box for each contour
    bboxs = [cv.boundingRect(contour) for contour in contours]
    # get roi for each bounding box, expand the roi by n pixels
    for (x, y, w, h) in bboxs:
        y1=y - npixel
        y2=y + h + npixel
        x1=x - npixel
        x2=x + w + npixel
        if y1<0:
            y1=0
            y2=h
        if y2>ori_img.shape[0]:
            y2=ori_img.shape[0]
            y1=y2-h
        if x1<0:
            x1=0
            x2=w
        if x2>ori_img.shape[1]:
            x2=ori_img.shape[1]
            x1=x2-w
        ori_rois.append(ori_img[y1:y2, x1:x2])
        label_rois.append(label_img[y1:y2, x1:x2])

    return ori_rois,label_rois,bboxs

def extract_roi_from_img(ori_img,label_img,top_npixel,bottom_npixel):
    gray=label_img.copy()
    ori_rois=[]
    label_rois=[]
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # get bounding box for each contour
    bboxs = [cv.boundingRect(contour) for contour in contours]
    # get roi for each bounding box, expand the roi by n pixels
    for (x, y, w, h) in bboxs:
        y1=y - top_npixel
        y2=y + h + bottom_npixel
        x1=0
        x2=ori_img.shape[1]
        if y1<0:
            y1=0
            y2=h
        if y2>ori_img.shape[0]:
            y2=ori_img.shape[0]
            y1=y2-h
        # if x1<0:
        #     x1=0
        #     x2=w
        # if x2>ori_img.shape[1]:
        #     x2=ori_img.shape[1]
        #     x1=x2-w

        ori_rois.append(ori_img[y1:y2, x1:x2])
        label_rois.append(label_img[y1:y2, x1:x2])

    return ori_rois,label_rois,bboxs

def transform(objs):
    return objs

# random color, exclude white and black
def random_color():
    bgr = np.random.randint(0, 255, size=(3,)).tolist()
    while bgr == [0,0,0] or bgr == [255,255,255]:
        bgr = np.random.randint(0, 255, size=(3,)).tolist()

    return bgr

# split the images into multiple grids, and fill the grid with random color
def fill_random_color(black,m,n):
    # get the height and width of the black image
    h, w = black.shape[:2]
    # create a random color image with the same size of the black image
    seg_img = np.zeros((h, w, 3), dtype=np.uint8)
    # fill the color image with the random color
    # get width and height of seg_img
    H, W = seg_img.shape[:2]
    for y in range(0,H,H//m):
        for x in range(0,W,W//n):
            x1=x
            x2=x+W//n
            y1=y
            y2=y+H//m
            if x2>W:
                x2=W
            if y2>H:
                y2=H
            # create a random color
            bgr = random_color()
            seg_img[y1:y2,x1:x2]=bgr
    # show seg_img
    # cv.imshow('content_seg', content_seg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return seg_img

if __name__ == '__main__':
    # read image
    img=cv.imread('D:\AI\colab\Texture Reformer\main.py\inputs\sdd\part29\style-transfer_test4\cont.jpg',0)
    out=fill_random_color(img,2,3)
    # show out img
    cv.imshow('out', out)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('D:\AI\colab\Texture Reformer\main.py\inputs\sdd\part29\style-transfer_test4\seg.png',out)
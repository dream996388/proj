import PIL
import numpy as np
import cv2 as cv
from PIL import Image


def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def to_indeximg(gray, cmap):
    gray = Image.fromarray(gray)
    img_P = gray.convert('P')
    img_P.putpalette((cmap * 255).astype(np.uint8).flatten())
    # return np.asarray(img_P)
    return img_P


# gray = cv.imread('D:\AI\colab\proj\issue\data_annotated\output_label.png', 0)
# # gray=Image.open('D:\AI\colab\proj\issue\data_annotated\output_label.png')
# colormap = label_colormap(255)
# index_img = to_indeximg(gray, colormap)
# index_img.save(r'D:\AI\colab\proj\issue\data_annotated\test_output_label.png')


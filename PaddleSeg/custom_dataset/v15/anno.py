import json

import numpy as np
from PIL import Image
import cv2 as cv
import base64
from labelme import LabelFile

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

def get_data(img_path):
    imgBytes = LabelFile.load_image_file(img_path)
    return base64.b64encode(imgBytes).decode('utf-8')

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
"""
data input and output.
"""

from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import numpy as np
import re
import sys
import math
import cv2
# import gdal
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import matplotlib.pyplot as plt


# def Read_Img_Tone(path, x_lu, y_lu, x_size, y_size):
#     dataset = gdal.Open(path)
#     if dataset == None:
#         print("GDAL RasterIO Error: Opening" + path + " failed!")
#         return
#
#     img = dataset.ReadAsArray(x_lu, y_lu, x_size, y_size)
#     img = np.swapaxes(img, 0, 2)
#     img = np.swapaxes(img, 0, 1)
#
#     del dataset
#
#     return img
#
#
# def GetSize(path):
#     dataset = gdal.Open(path)
#     if dataset == None:
#         print("GDAL RasterIO Error: Opening" + path + " failed!")
#         return
#
#     width = dataset.RasterXSize
#     height = dataset.RasterYSize
#
#     del dataset
#     return width, height


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def write_cam(file, cam, location):
    f = open(file, "w")
    # f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    for word in location:
        f.write(str(word) + ' ')
    f.write('\n')

    f.close()


def load_pfm(fname):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data


def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale))

    image_string = image.tostring()
    file.write(image_string)

    file.close()


class RandomCrop(object):
    def __init__(self, CropSize=0.1):
        self.CropSize = CropSize

    def __call__(self, image, normal):
        import random, cv2
        h, w = normal.shape[:2]
        img_h, img_w = image.shape[:2]
        CropSize_w, CropSize_h = max(1, int(w * self.CropSize)), max(1, int(h * self.CropSize))
        x1, y1 = random.randint(0, CropSize_w), random.randint(0, CropSize_h)
        x2, y2 = random.randint(w - CropSize_w, w), random.randint(h - CropSize_h, h)

        normal_crop = normal[y1:y2, x1:x2]
        normal_resize = cv2.resize(normal_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        image_crop = image[4*y1:4*y2, 4*x1:4*x2]
        image_resize = cv2.resize(image_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        return image_resize, normal_resize


if __name__ == '__main__':

    depth1, scale1 = read_pfm('C:/Users/liujin/Desktop/新建文件夹 (2)/0.25_0.25/000005_init.pfm')
    depth2, scale2 = read_pfm('C:/Users/liujin/Desktop/新建文件夹 (2)/0_0/000005_init.pfm')
    diff = depth1 - depth2
    cv2.imwrite("C:/Users/liujin/Desktop/casrednet/diff_000005_init_0.25_0.tif", diff)
    print(diff)
    """cv2.imshow("diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""


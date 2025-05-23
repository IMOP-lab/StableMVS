"""
data preprocesses.
"""

from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    # distortion
    # new_cam[1][0][1] = cam[1][0][1] * scale
    # new_cam[1][1][0] = cam[1][1][0] * scale
    return new_cam


def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(len(cams)):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    #h, w = image.shape[0:2]
    #new_w = int(w * scale)
    #new_h = int(h * scale)
    if interpolation == 'linear':
        #return image.resize((new_h, new_w), Image.BILINEAR)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'biculic':
        #return image.resize((new_h, new_w), Image.BICUBIC)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def scale_input(image, cam, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    image = scale_image(image, scale=scale)
    cam = scale_camera(cam, scale=scale)
    if depth_image is None:
        return image, cam
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='linear')
        return image, cam, depth_image


def crop_input(image, cam, depth_image=None, max_h=384, max_w=768, resize_scale=1, base_image_size=32):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
    max_h = int(max_h * resize_scale)
    max_w = int(max_w * resize_scale)
    h, w = image.shape[0:2]
    new_h = h
    new_w = w
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.ceil(h / base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.ceil(w / base_image_size) * base_image_size)
    # start_h = int(math.ceil((h - new_h) / 2))
    # start_w = int(math.ceil((w - new_w) / 2))
    start_h = 0
    start_w = 0
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    image = image[start_h:finish_h, start_w:finish_w]
    cam[1][0][2] = cam[1][0][2] - start_w
    cam[1][1][2] = cam[1][1][2] - start_h

    # crop depth image
    if depth_image is not None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return image, cam, depth_image
    else:
        return image, cam


def center_image(img):
    # scale 0~255 to 0~1
    """np_img = np.array(img, dtype=np.float32) / 255.
    return np_img"""
    # normalize image input
    img_array = np.array(img)
    img = img_array.astype(np.float32)
    # img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


######### old version read camera files   ############
def read_cam_meitan(file, ndepths, interval_scale=1):
    """ read camera txt file  (XrightYdown，[Rcw|twc])"""
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extrinsics[i][j] = words[extrinsic_index]  # Rcw|twc

    # trans Rcw to Rwc
    R = extrinsics[0:3, 0:3]
    extrinsics[0:3, 0:3] = np.linalg.inv(R)
    # Twc to Tcw
    extrinsics = np.linalg.inv(extrinsics)
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    # K Photogrammetry system XrightYdown
    f = pera[0][0]
    x0 = pera[0][1]
    y0 = pera[0][2]
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


def read_cam_whu(file, ndepths, interval_scale=1):
    """ read camera txt file (XrightYup，Twc)"""
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extrinsics[i][j] = words[extrinsic_index]  # Twc

    # if cam ori is XrightYup
    O = np.eye(3, dtype=np.float32)
    O[1, 1] = -1
    O[2, 2] = -1
    R = extrinsics[0:3, 0:3]
    R2 = np.matmul(R, O)
    extrinsics[0:3, 0:3] = R2

    extrinsics = np.linalg.inv(extrinsics) # Tcw
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    # K
    f = pera[0][0]
    x0 = pera[0][1]
    y0 = pera[0][2]
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


def read_cam_tianjin(file, ndepths, interval_scale=1):
    """ read camera txt file (XrightYup，Twc)"""
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extrinsics[i][j] = words[extrinsic_index]  # Twc

    # if cam ori is XrightYup
    O = np.eye(3, dtype=np.float32)
    O[1, 1] = -1
    O[2, 2] = -1
    R = extrinsics[0:3, 0:3]
    R2 = np.matmul(R, O)
    extrinsics[0:3, 0:3] = R2
    extrinsics[0][3] = extrinsics[0][3]-490000.0
    extrinsics[1][3] = extrinsics[1][3]-4310000.0
    extrinsics[2][3] = extrinsics[2][3]

    extrinsics = np.linalg.inv(extrinsics) # Tcw
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    f = pera[0][0]
    x0 = pera[0][1]
    y0 = pera[0][2]
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


def read_sat_file(file, ndepths, interval_scale=1):
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j
            extrinsics[i][j] = words[extrinsic_index]  # Tcw
    #extrinsics = np.linalg.inv(extrinsics) # Tcw
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[16 + i]


    f = pera[0][0]
    x0 = pera[0][1]
    y0 = pera[0][2]
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    cam[1][3][2] = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32
    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


def tr_read_cam_munchen(file, ndepths, interval_scale=1):
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extrinsics[i][j] = words[extrinsic_index]  # Twc

    # if cam ori is XrightYup
    O = np.eye(3, dtype=np.float32)
    O[1, 1] = -1
    O[2, 2] = -1
    R = extrinsics[0:3, 0:3]
    R2 = np.matmul(R, O)
    extrinsics[0:3, 0:3] = R2

    extrinsics = np.linalg.inv(extrinsics) # Tcw
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    f = pera[0][0]
    x0 = pera[0][1] - pera[0][7] - (pera[0][10] * pera[0][11])  # munchen
    y0 = pera[0][2] - pera[0][8] - (pera[0][9] * pera[0][12])
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


def tr_read_cam_whu(file, ndepths, interval_scale=1):
    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float32)
    extrinsics = np.zeros((4, 4), dtype=np.float32)
    pera = np.zeros((1, 13), dtype=np.float32)
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            extrinsics[i][j] = words[extrinsic_index]  # Twc

    # if cam ori is XrightYup
    O = np.eye(3, dtype=np.float32)
    O[1, 1] = -1
    O[2, 2] = -1
    R = extrinsics[0:3, 0:3]
    R2 = np.matmul(R, O)
    extrinsics[0:3, 0:3] = R2

    extrinsics = np.linalg.inv(extrinsics) # Tcw
    cam[0, :, :] = extrinsics

    for i in range(0, 13):
        pera[0][i] = words[17 + i]

    f = pera[0][0]
    # x0 = pera[0][1] - pera[0][7] - (pera[0][10] * pera[0][11])  # munchen
    # y0 = pera[0][2] - pera[0][8] - (pera[0][9] * pera[0][12])
    x0 = pera[0][1]  # whu
    y0 = pera[0][2]
    cam[1][0][0] = f
    cam[1][1][1] = f
    cam[1][0][2] = x0
    cam[1][1][2] = y0
    cam[1][2][2] = 1

    # depth range
    cam[1][3][0] = np.float32(pera[0][3])  # start
    cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
    cam[1][3][3] = np.float32(pera[0][4])  # end

    acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

    if acturald > ndepths:
        scale = acturald / np.float32(ndepths)
        cam[1][3][1] = cam[1][3][1] * scale
        acturald = ndepths
    cam[1][3][2] = acturald
    #cam[1][3][2] = ndepths
    location = words[23:30]

    return cam, location


# For training
def gen_train_mvs_list(data_folder, view_num, fext='.png', mode='training'):
    """ generate data paths for whu dataset """
    sample_list = []
    
    # parse camera pairs
    cluster_file_path = data_folder + '/pair.txt'
    cluster_list = open(cluster_file_path).read().split()

    # 3 sets
    train_cluster_path = data_folder + '/index.txt'
    training_set = open(train_cluster_path).read().split()

    data_set = []
    if mode == 'training':
        data_set = training_set

    # for each dataset
    for i in data_set:
        image_folder = os.path.join(data_folder, ('Images/%s' % i)).replace("\\", "/")
        cam_folder = os.path.join(data_folder, ('Cams/%s' % i)).replace("\\", "/")
        depth_folder = os.path.join(data_folder, ('Depths/%s' % i)).replace("\\", "/")

        if mode == 'training':
            # for each view
            for p in range(0, int(cluster_list[0])): # 0-4
                index_ref = int(cluster_list[(int(cluster_list[0])+1) * p + 1])
                image_folder2 = os.path.join(image_folder, ('%d' % index_ref)).replace("\\", "/")
                image_files = sorted(os.listdir(image_folder2))

                for j in range(0, int(np.size(image_files))):
                    paths = []
                    portion = os.path.splitext(image_files[j])
                    newcamname = portion[0] + '.txt'
                    newdepthname = portion[0] + fext

                    # ref image
                    ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_ref)), image_files[j]).replace("\\", "/")
                    ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_ref)), newcamname).replace("\\", "/")
                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    # view images
                    for view in range(view_num - 1):
                        index_view = int(cluster_list[(int(cluster_list[0])+1) * p + 3 + view])  # selected view image
                        view_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_view)), image_files[j]).replace("\\", "/")
                        view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_view)), newcamname).replace("\\", "/")
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # depth path
                    depth_image_path = os.path.join(os.path.join(depth_folder, ('%d' % index_ref)), newdepthname).replace("\\", "/")
                    paths.append(depth_image_path)
                    sample_list.append(paths)

    return sample_list


# for testing
def gen_test_mvs_list(dense_folder, view_num, fext='.png'):
    """ mvs input path list """

    cluster_list_path = os.path.join(dense_folder, 'pair.txt')
    cluster_list = open(cluster_list_path).read().split()

     # test sets
    test_cluster_path = dense_folder + '/index.txt'
    test_set = open(test_cluster_path).read().split()

    # for each dataset
    mvs_list = []
    for m in test_set:
        image_folder = os.path.join(dense_folder, ('Images/%s' % m)).replace("\\", "/")
        cam_folder = os.path.join(dense_folder, ('Cams/%s' % m)).replace("\\", "/")
        depth_folder = os.path.join(dense_folder, ('Depths/%s' % m)).replace("\\", "/")

        for i in range(int(cluster_list[0])):# 0-4
            index_ref=int(cluster_list[(int(cluster_list[0])+1) * i + 1])
            image_folder2=os.path.join(image_folder, ('%d' % index_ref)).replace("\\", "/")
            image_files = sorted(os.listdir(image_folder2))  

            for j in range(0,int(np.size(image_files))):
                paths = []
                portion = os.path.splitext(image_files[j])   
                newcamname = portion[0] + '.txt'
                #newdepthname = portion[0] + '.png'
                newdepthname = portion[0] + fext

                # ref image
                ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_ref)), image_files[j]).replace("\\", "/")
                ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_ref)), newcamname).replace("\\", "/")
                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                # view images
                all_view_num = int(cluster_list[2])
                check_view_num = min(view_num - 1, all_view_num)
                for view in range(check_view_num):
                    index_view = int(cluster_list[(int(cluster_list[0])+1) * i + 3 + view]) # selected view image
                    view_image_path = os.path.join(os.path.join(image_folder, ('%d' % index_view)), image_files[j]).replace("\\", "/")
                    view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % index_view)), newcamname).replace("\\", "/")
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                # depth path
                ref_depth_path = os.path.join(os.path.join(depth_folder, ('%d' % index_ref)), newdepthname).replace("\\", "/")
                paths.append(ref_depth_path)
                mvs_list.append(paths)

    return mvs_list


# for predict without depth
def gen_predict_mvs_list(dense_folder, view_num, fext='.png'):
    """ mvs input path list """

     # 3 sets
    test_cluster_path = dense_folder + '/viewpair.txt'
    cluster_list = open(test_cluster_path).read().split()

    image_folder = os.path.join(dense_folder, 'Images').replace("\\", "/")
    cam_folder = os.path.join(dense_folder, 'Cams').replace("\\", "/")

    # for each dataset
    mvs_list = []
    total_num = int(cluster_list[0])
    all_view_num = int(cluster_list[1])

    for i in range(total_num):# 0-4
        paths = []
        index_ref = cluster_list[(all_view_num) * i * 2 + 2]  # reference
        ref_image_path = os.path.join(image_folder, '{}'.format(index_ref + fext)).replace("\\", "/")
        ref_cam_path = os.path.join(cam_folder, '{}.txt'.format(index_ref)).replace("\\", "/")
        paths.append(ref_image_path)
        paths.append(ref_cam_path)

        # view images
        check_view_num = min(view_num - 1, all_view_num)
        for view in range(check_view_num):
            index_view = cluster_list[(all_view_num) * i * 2 + 4 + view * 2]  # source
            view_image_path = os.path.join(image_folder, '{}'.format(index_view + fext)).replace("\\", "/")
            view_cam_path = os.path.join(cam_folder, '{}.txt'.format(index_view)).replace("\\", "/")
            paths.append(view_image_path)
            paths.append(view_cam_path)

        mvs_list.append(paths)

    return mvs_list


# for training with zy3 dataset
def gen_mvs_list(data_folder, view_num, mode='training'):
    """ generate data paths for zy3 dataset """
    sample_list = []

    for r in range(view_num):
        image_folder = os.path.join(data_folder, ('image/%s' % r)).replace("\\", "/")
        rpc_folder = os.path.join(data_folder, ('cameras/%s' % r)).replace("\\", "/")
        height_folder = os.path.join(data_folder, ('depth/%s' % r)).replace("\\", "/")

        image_files = os.listdir(image_folder)

        for p in image_files:
            sample = []

            name = os.path.splitext(p)[0]
            ref_image = os.path.join(image_folder, '{}.png'.format(name)).replace("\\", "/")
            ref_cam = os.path.join(rpc_folder, '{}.txt'.format(name)).replace("\\", "/")
            ref_height = os.path.join(height_folder, '{}.pfm'.format(name)).replace("\\", "/")

            sample.append(ref_image)
            sample.append(ref_cam)

            for s in range(view_num):
                sv = (r + s) % view_num

                if sv != r:
                    source_image = os.path.join(data_folder, 'image/{}/{}.png'.format(sv, name)).replace("\\", "/")
                    source_rpc = os.path.join(data_folder, 'cameras/{}/{}.txt'.format(sv, name)).replace("\\", "/")

                    sample.append(source_image)
                    sample.append(source_rpc)
            sample.append(ref_height)

            sample_list.append(sample)

    return sample_list


# For training with zy3 dataset (reference view)
def gen_mvs_list_refview(data_folder, view_num, ref_view=2):
    """ generate data paths for zy3 dataset """
    sample_list = []

    image_folder = os.path.join(data_folder, ('image/%s' % ref_view)).replace("\\", "/")
    rpc_folder = os.path.join(data_folder, ('cameras/%s' % ref_view)).replace("\\", "/")
    height_folder = os.path.join(data_folder, ('depth/%s' % ref_view)).replace("\\", "/")

    image_files = os.listdir(image_folder)

    for p in image_files:
        sample = []

        name = os.path.splitext(p)[0]
        ref_image = os.path.join(image_folder, '{}.png'.format(name)).replace("\\", "/")
        ref_camera = os.path.join(rpc_folder, '{}.txt'.format(name)).replace("\\", "/")
        ref_depth = os.path.join(height_folder, '{}.pfm'.format(name)).replace("\\", "/")

        sample.append(ref_image)
        sample.append(ref_camera)

        for s in range(view_num):
            sv = (ref_view + s) % view_num

            if sv != ref_view:
                source_image = os.path.join(data_folder, 'image/{}/{}.png'.format(sv, name)).replace("\\", "/")
                source_camera = os.path.join(data_folder, 'cameras/{}/{}.txt'.format(sv, name)).replace("\\", "/")

                sample.append(source_image)
                sample.append(source_camera)
        sample.append(ref_depth)

        sample_list.append(sample)

    return sample_list


# data augment
def image_augment(image):

    image = randomColor(image)
    #image = randomGaussian(image, mean=0.2, sigma=0.3)

    return image
    

def randomColor(image):

    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image


def randomGaussian(image, mean=0.02, sigma=0.03):

    def gaussianNoisy(im, mean=0.02, sigma=0.03):


        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])

    return Image.fromarray(np.uint8(img))
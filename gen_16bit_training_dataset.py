#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import os
import argparse

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Face Class Datasets.')

    parser.add_argument('-i', '--img_in_dir', default=r'../datasets/data_collect/IR/', type=str,
                        help='folder dir to input image')
    parser.add_argument('-o', '--img_out_dir', default=r'/data/tomzhai/face_align/datasets/training_materials/image/bbox', type=str,
                        help='folder dir to output image')
    parser.add_argument('-t', '--txt_out_path',
                        default=r'/data/tomzhai/face_align/datasets/training_materials/txt/data_collect_labels.txt', type=str,
                        help='folder dir to output image')

    args = parser.parse_args()
    img_in_dir = args.img_in_dir
    img_out_dir = args.img_out_dir
    img_list = getFiles(img_in_dir, 'jpg')
    txt_path = args.txt_out_path
    f = open(txt_path, 'w')
    for _i, img_path in enumerate(img_list):
        frame = cv2.imread(img_path)
        box_bin_path = img_path.replace('img', 'box')
        box_bin_path = box_bin_path.replace('IR', 'IR_BIN')
        box_bin_path = box_bin_path.replace('jpg', 'bin')
        eular_bin_path = img_path.replace('img', 'eular')
        eular_bin_path = eular_bin_path.replace('IR', 'IR_BIN')
        eular_bin_path = eular_bin_path.replace('jpg', 'bin')
        box_bin = np.fromfile(box_bin_path, 'float64')
        eular_bin = np.fromfile(eular_bin_path, 'float64')
        img_name = "data_collect_img_%06d.jpg" % _i
        img_out_path = os.path.join(img_out_dir, img_name)
        f.write(img_name + " ")
        f.write("%f %f %f\n" % (eular_bin[0], eular_bin[1], eular_bin[2]))
        b = list(map(int, box_bin))
        cv2.imwrite(img_out_path, frame[b[1]:b[3], b[0]:b[2]])
        print(_i)
    f.close()



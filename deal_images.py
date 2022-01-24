#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import sys
import cv2
import service
import os
import argparse

fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.95)
fa = service.DenseFaceReconstruction("weights/sparse_face.tflite")
mr = service.TrianglesMeshRender("asset/render.so", "asset/triangles.npy")

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(suffix)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Face Class Datasets.')

    parser.add_argument('-i', '--img_in_dir', default=r'/data/public/datasets/face_anti_spoofing/datasets/data_collect', type=str,
                        help='folder dir to input image')
    parser.add_argument('-o', '--img_out_dir', default=r'../dealed_yimaitong2/image/st', type=str,
                        help='folder dir to output image')

    args = parser.parse_args()
    img_in_dir = args.img_in_dir
    img_out_dir = args.img_out_dir
    img_list = getFiles(img_in_dir, 'bmp')
    for _i, img_path in enumerate(img_list):
        frame = cv2.imread(img_path)

        boxes, scores = fd.inference(frame)
        feed = frame.copy()
        results = fa.get_landmarks(feed, boxes)
        if len(results.gi_frame.f_locals['detected_faces']) > 1:
            continue
        for results in fa.get_landmarks(feed, boxes):
            service.sparse(frame, results, color=(224, 255, 255))

        results = fa.get_landmarks(feed, boxes)
        if len(results.gi_frame.f_locals['detected_faces']) > 1:
            continue
        for results in fa.get_landmarks(feed, boxes):
            eular = service.pose(frame, results, color=(224, 255, 255))


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
    parser.add_argument('-o', '--img_out_dir', default=r'../datasets/data_collect', type=str,
                        help='folder dir to output image')

    args = parser.parse_args()
    img_in_dir = args.img_in_dir
    img_out_dir = args.img_out_dir
    img_list = getFiles(img_in_dir, 'raw')
    _cnt = 0
    RGB_txt_path = os.path.join(img_out_dir, 'RGB_TXT/rgb_file_name.txt')
    f_RGB = open(RGB_txt_path, 'w')
    IR_txt_path = os.path.join(img_out_dir, 'IR_TXT/ir_file_name.txt')
    f_IR = open(IR_txt_path, 'w')
    for _i, img_path in enumerate(img_list):
        if "real" not in img_path:
            continue
        mode_str = ""
        if "IR" in img_path:
            ir_data = np.fromfile(img_path, dtype=np.uint16)
            ir_data = ir_data.reshape(640, -1)
            ir_data = (np.right_shift(ir_data, 2) & 0xff).astype(np.uint8)
            frame = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2BGR)
            mode_str = "IR"
        elif "RGB" in img_path:
            continue
            frame = cv2.imread(img_path)
            mode_str = "RGB"
        else:
            continue
        boxes, scores = fd.inference(frame)
        if len(boxes) != 1:
            continue
        box_bin_saved_name = mode_str + "_BIN/box_%06d.bin" % _cnt
        landmark_bin_saved_name = mode_str + "_BIN/landmark_%06d.bin" % _cnt
        eular_bin_saved_name = mode_str + "_BIN/eular_%06d.bin" % _cnt
        np.reshape(boxes, [1, -1])[0].tofile(os.path.join(img_out_dir, box_bin_saved_name))
        feed = frame.copy()
        for results in fa.get_landmarks(feed, boxes):
            landmarks = service.sparse(frame, results, color=(224, 255, 255))
        np.reshape(landmarks, [1, -1])[0].tofile(os.path.join(img_out_dir, landmark_bin_saved_name))

        for results in fa.get_landmarks(feed, boxes):
            eular = service.pose(frame, results, color=(224, 255, 255))
        np.reshape(eular, [1, -1])[0].tofile(os.path.join(img_out_dir, eular_bin_saved_name))

        img_name = mode_str + "/img_%06d.jpg" % _cnt
        print(_cnt, _i)
        img_out_path = os.path.join(img_out_dir, img_name)
        cv2.imwrite(img_out_path, frame)
        _cnt += 1
        if "IR" in img_path:
            f_IR.write(img_path + " " + img_out_path + "\n")
        elif "RGB" in img_path:
            f_RGB.write(img_path + " " + img_out_path + "\n")
        else:
            continue
    f_RGB.close()
    f_IR.close()




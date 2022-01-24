#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import sys
import cv2
import service

fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.52)
fa = service.DenseFaceReconstruction("weights/sparse_face.tflite")
# mr = service.TrianglesMeshRender("asset/render.so", "asset/triangles.npy")

frame = cv2.imread(sys.argv[1])

boxes, scores = fd.inference(frame)
feed = frame.copy()
handler = getattr(service, "sparse")
for results in fa.get_landmarks(feed, boxes):
    handler(frame, results, color=(224, 255, 255))
cv2.imwrite('./image_out/test_sparse_out.jpg', frame)

# handler = getattr(service, "pose")
for results in fa.get_landmarks(feed, boxes):
    eular = service.pose(frame, results, color=(224, 255, 255))
cv2.imwrite('./image_out/test_pose_out.jpg', frame)

# cv2.imshow("result", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

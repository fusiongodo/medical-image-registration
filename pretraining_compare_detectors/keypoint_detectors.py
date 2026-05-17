import numpy as np
import cv2
import torch

from external.SuperPointPretrainedNetwork import SuperPointFrontend


class FastDetector:
    name = "FAST"

    def __init__(self, threshold=20, nonmax_suppression=True):
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression,
        )

    def detect(self, image_gray):
        img_u8 = (image_gray * 255).clip(0, 255).astype(np.uint8)
        keypoints = self.detector.detect(img_u8, None)

        if not keypoints:
            return np.zeros((3, 0), dtype=np.float32)

        return np.array(
            [[kp.pt[0], kp.pt[1], kp.response] for kp in keypoints],
            dtype=np.float32,
        ).T


class SuperPointDetector:
    name = "SuperPoint"

    def __init__(
        self,
        weights_path="../external/superpoint_v1.pth",
        nms_dist=8,
        conf_thresh=0.015,
        nn_thresh=0.7,
        cuda=None,
    ):
        if cuda is None:
            cuda = torch.cuda.is_available()

        self.superpoint = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=cuda,
        )

    def detect(self, image_gray):
        pts, desc, heatmap = self.superpoint.run(image_gray)
        return pts
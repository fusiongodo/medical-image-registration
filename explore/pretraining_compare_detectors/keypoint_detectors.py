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


class HarrisDetector:
    name = "Harris"

    def __init__(self, block_size=2, ksize=3, k=0.04, threshold=0.01):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold = threshold

    def detect(self, image_gray):
        img_u8 = (image_gray * 255).clip(0, 255).astype(np.uint8)
        response = cv2.cornerHarris(img_u8, self.block_size, self.ksize, self.k)
        mask = response > self.threshold * response.max()
        ys, xs = np.where(mask)

        if len(xs) == 0:
            return np.zeros((3, 0), dtype=np.float32)

        scores = response[ys, xs]
        return np.array([xs, ys, scores], dtype=np.float32)


class ShiTomasiDetector:
    name = "Shi-Tomasi"

    def __init__(self, max_corners=500, quality_level=0.01, min_distance=5):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

    def detect(self, image_gray):
        img_u8 = (image_gray * 255).clip(0, 255).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(
            img_u8,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )

        if corners is None:
            return np.zeros((3, 0), dtype=np.float32)

        corners = corners.reshape(-1, 2)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        return np.hstack([corners, ones]).T


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
        if pts.ndim != 2 or pts.shape[0] != 3:
            return np.zeros((3, 0), dtype=np.float32)
        return pts
#!/usr/bin/env python3

import os
import numpy as np
import glob
import scipy.io

# from lib.utils import logger

# Joint index to label map for the SLP Dataset.
SLPDatasetJointToLabels = {
    0: "Right ankle",
    1: "Right knee",
    2: "Right hip",
    3: "Left hip",
    4: "Left knee",
    5: "Left ankle",
    6: "Right wrist",
    7: "Right elbow",
    8: "Right shoulder",
    9: "Left shoulder",
    10: "Left elbow",
    11: "Left wrist",
    12: "Thorax",
    13: "Head top",
}

# Label to joint index map for the SLP Dataset.
SLPDatasetLabelsToJoint = {v: k for k, v in SLPDatasetJointToLabels.items()}

# Indexes of Left-Right joint pairs. Useful for augmentation where right limbs
# need to be swapped with left limbs. The first index is the original left
# joint, and the second index is the original right joint.
SLPDatasetLeftRightJointPairs = {
    "shoulder": [9, 8],
    "elbow": [10, 7],
    "wrist": [11, 6],
    "hip": [3, 2],
    "knee": [4, 1],
    "ankle": [5, 0],
}


def loadImagePathsAndLabels(dataDir, onlyAnnotated=False):
    imgDirs = sorted(glob.glob(os.path.join(dataDir, "*")))

    allImgPaths = []
    allKeyPts = {}
    for d in imgDirs:
        # Checking if labels exist
        labelsPath = os.path.join(d, "joints_gt_IR.mat")
        labelsExist = os.path.exists(labelsPath)
        if not labelsExist and onlyAnnotated:
            print(f"skipping directory without annotations: {d}")
            continue

        # Getting image paths
        imgPaths = sorted(glob.glob(os.path.join(d, "IR", "**", "*.png")))
        numImg = len(imgPaths)

        keyPts = None
        numLabels = 0
        if labelsExist:
            # Loading key points array. The array has shape (3, 14, 45) where 14 =
            # number of joints, 45 = number of images per directory and 3 = x and y
            # coordinates + occlusion label.
            keyPts = scipy.io.loadmat(labelsPath)['joints_gt']
            numLabels = keyPts.shape[-1]

            if numLabels != numImg:
                logger.error(f"number of images and labels are not equal: {numImg} vs {numLabels}")
                continue

            # Adding labels
            for idx, path in enumerate(imgPaths):
                joints = np.float32(keyPts[:, :, idx])
                allKeyPts[path] = joints.T  # transposing => (numJoints, coordinates)

        # Adding image paths
        allImgPaths.extend(imgPaths)

    return allImgPaths, allKeyPts

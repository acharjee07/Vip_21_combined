#!/usr/bin/env python3

import numpy as np


def calcPCKh(gt, pred, th):
    head_size = np.linalg.norm(gt[12] - gt[13])
    dist_threshold = head_size * th
    distances = np.linalg.norm(gt - pred, axis=1)
    pckh = np.sum(distances < dist_threshold) / gt.shape[0]
    return pckh


def calcPCKhBatch(gt_batch, pred_batch, th):
    batch_pckh = []
    for x in range(len(gt_batch)):
        gt = gt_batch[x]
        pred = pred_batch[x]
        batch_pckh.append(calcPCKh(gt, pred, th=th))
    return np.mean(batch_pckh)

def calcAllPCKhBatch(gt_batch, pred_batch, th):
    batch_pckh = []
    batch_all_pckh = np.zeros(14)
    for x in range(len(gt_batch)):
        #Sliced gt and pred for indivisual images
        gt = gt_batch[x]
        pred = pred_batch[x]

        #determined the DISTANCE_TRESHOLD and DISTANCE value between gt and predicted keypoints 
        head_size = np.linalg.norm(gt[12] - gt[13])
        dist_threshold = head_size * th
        distances = np.linalg.norm(gt - pred, axis=1)  

        #Checked with the dist filter
        dist_filt = (distances < dist_threshold)

        #added average of PCKh of all keypoint to the list
        pckh = np.sum(dist_filt) / gt.shape[0]
        batch_pckh.append(pckh)

        #Sum of keypointwise PCKh values
        batch_all_pckh += dist_filt*1

    #average of keypointwise PCKh VAlues
    batch_all_pckh = batch_all_pckh / (x+1)
    
    return np.mean(batch_pckh), batch_all_pckh

# print(calcAllPCKhBatch(np.random.rand(3,14,2), np.random.rand(3,14,2), th=.5))

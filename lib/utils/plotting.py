#!/usr/bin/env python3

import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

colorRGBMap = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "green": (0, 255, 0),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
}


def drawJoints(img, keyPts, color=(0, 255, 0), thickness=2):
    joints = keyPts.copy()
    if np.max(keyPts) <= 1.0:
         # multiply (x,y) by (width/2, height/2) and add (width/2, height/2)
        centerX = img.shape[1] / 2
        centerY = img.shape[0] / 2
        joints[:, 0] = (joints[:, 0] * centerX) + centerX
        joints[:, 1] = (joints[:, 1] * centerY) + centerY
    joints = joints.astype(int)

    if isinstance(color, str):
        color = color.lower()
        if color.startswith("#"):
            color = colorHex2RGB(color)
        else:
            color = colorRGBMap[color]

    img2 = img.copy()
    img2 = cv2.line(img2, tuple(joints[0]), tuple(joints[1]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[1]), tuple(joints[2]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[3]), tuple(joints[4]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[4]), tuple(joints[5]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[6]), tuple(joints[7]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[7]), tuple(joints[8]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[9]), tuple(joints[10]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[10]), tuple(joints[11]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[9]), tuple(joints[12]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[8]), tuple(joints[12]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[2]), tuple(joints[3]), color, thickness)
    img2 = cv2.line(img2, tuple(joints[12]), tuple(((joints[2] + joints[3]) / 2).astype(int)), color, thickness)
    img2 = cv2.line(img2, tuple(joints[12]), tuple(joints[13]), color, thickness)

    return img2


def colorHex2RGB(hexCode):
    return tuple(int(hexCode[i:i + 2], 16) for i in (0, 2, 4))


def getSquarePlotLayout(numPlots):
    """
    Returns the number of rows and columns for a good
    square-ish layout for the given number of subplots.

    Parameters
    ----------
    numPlots : int
        Number of subplots

    Returns
    -------
    int, int
        Number of rows and columns in layout.
    """
    nRows = int(np.round(np.sqrt(numPlots)))
    nCols = 1
    while nRows * nCols < numPlots:
        nCols += 1

    return nRows, nCols


def convertToTfTensor(figure):
    """
    Converts given matplotlib.Figure into a Tensorflow
    Tensor that can be written to Tensorboard.

    Parameters
    ----------
    figure : matplotlib.Figure
        Matplotlib Figure handle.

    Returns
    -------
    tf.Tensor
        Tensor containing the figure as image data.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def combineHeatmapChannels(hmap):
    """
    Returns the union of all the channels in the heatmap
    as a single 2D map.

    Parameters
    ----------
    hmap : np.array, shape = (Y, X, numJoints)
        Heatmap with separate joints along
        channel dimension

    Returns
    -------
    np.array, shape = (Y, X)
        Heatmap created from union of all
        channels.
    """
    return np.squeeze(np.max(hmap, axis=-1))


def plotHeatmaps(img, label, pred, title=None, returnAsTensor=False):
    """
    Plots the ground truth and predicted heatmaps overlayed
    on top of the image. The plot is either returned as a
    tensorflow image tensor or a matplotlib figure.

    Parameters
    ----------
    img : np.array, shape = (Y, X, Ch)
        Image on which heatmaps will be overlayed.
    label : np.array, shape = (Y, X, numJoints)
        Ground truth heatmaps.
    pred : np.array, shape = (Y, X, numJoints)
        Predicted heatmaps.
    title : str, optional
        Title for the plot, by default None.
    returnAsTensor : bool, optional
        If True, plot will be returned as a
        tensorflow image tensor. Otherwise the plot
        returned as a matplotlib figure.
        By default True.

    Returns
    -------
    tensorflow.Tensor or matplotlib.Figure
        Image with heatmaps overlayed.
    """
    fig = plt.figure(figsize=(16, 8))

    maxVal = np.max(img)
    if maxVal <= 1.0:
        img = np.clip(img, 0, 1.0)
    else:
        img = np.uint8(np.clip(img, 0, 255))

    # Resizing image if required to match heatmap resolution for plotting.
    heatmapRes = label.shape[:2]
    imageRes = img.shape[:2]
    if heatmapRes != imageRes:
        img = cv2.resize(img, (heatmapRes[1], heatmapRes[0]))

    if label is not None:
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.imshow(img)
        plt.imshow(combineHeatmapChannels(label), alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.title("Predicted")
        plt.imshow(img)
        plt.imshow(combineHeatmapChannels(pred), alpha=0.5)

        plt.tight_layout()
    else:
        plt.subplot(1, 1, 1)
        plt.title("Predicted")
        plt.imshow(img)
        plt.imshow(combineHeatmapChannels(pred))

    if title is not None:
        plt.suptitle(title)

    if not returnAsTensor:
        return fig

    return convertToTfTensor(fig)


def plotHeatmapChannels(img, heatmap, channelLabels=None, title=None, returnAsTensor=False):
    """
    Plots the given heatmap channels individually overlayed
    on top of the image. The plot is either returned as a
    tensorflow image tensor or a matplotlib figure.

    Parameters
    ----------
    img : np.array, shape = (Y, X, Ch)
        Image on which heatmaps will be overlayed.
    heatmap : np.array, shape = (Y, X, numJoints)
        Heatmap to plot.
    channelLabels : dict { int : str } or list of str, optional
        Map / List of labels for each channel of the heatmap
        which will be included on each subplot. If given as a map,
        the keys are the channel indexes (0-based). 
    title : str, optional
        Title for the plot, by default None.
    returnAsTensor : bool, optional
        If True, plot will be returned as a
        tensorflow image tensor. Otherwise the plot
        returned as a matplotlib figure.
        By default True.

    Returns
    -------
    tensorflow.Tensor or matplotlib.Figure
        Image with heatmap channels overlayed separately.
    """
    numChannels = heatmap.shape[-1]
    if channelLabels is not None:
        if len(channelLabels) != numChannels:
            raise ValueError("number of channel labels not equal to number of heatmap channels")

        if isinstance(channelLabels, list):
            channelLabels = {idx: label for idx, label in enumerate(channelLabels)}

    fig = plt.figure(figsize=(16, 8))

    maxVal = np.max(img)
    if maxVal <= 1.0:
        img = np.clip(img, 0, 1.0)
    else:
        img = np.uint8(np.clip(img, 0, 255))

    # Resizing image if required to match heatmap resolution for plotting.
    heatmapRes = heatmap.shape[:2]
    imageRes = img.shape[:2]
    if heatmapRes != imageRes:
        img = cv2.resize(img, (heatmapRes[1], heatmapRes[0]))

    nRows, nCols = getSquarePlotLayout(numChannels)
    for idx in range(numChannels):
        plt.subplot(nRows, nCols, idx + 1)
        plt.imshow(img)
        plt.imshow(heatmap[..., idx], alpha=0.5)

        if channelLabels is not None:
            plt.title(channelLabels[idx])

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    if not returnAsTensor:
        return fig

    return convertToTfTensor(fig)

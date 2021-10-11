#!/usr/bin/env python3

import numpy as np




def heatmap2keypts(heatmap):
    """
    Converts given heatmap or batch of heatmaps into x and y coordinates of joints.

    Parameters
    ----------
    heatmap : np.array, shape = (Y, X, numJoints) or (Batch, Y, X, numJoints)
        Heatmap of different joints.

    Returns
    -------
    np.array, shape = (numJoints, 2) or (Batch, Y, X, numJoints)
        Array containing x and y coordinates for all joints.
    """
    # TODO: (shahruk) figure out post processing to refine conversion.
    # Check https://github.com/anibali/pytorch-stacked-hourglass/blob/master/src/stacked_hourglass/utils/evaluation.py#L10

    nDims = len(heatmap.shape)
    if nDims < 3 or nDims > 4:
        raise RuntimeError("expected heatmaps to have either 3 or 4 (batched) dimensions")

    if nDims == 3:
        return heatmap2keyptsSingle(heatmap)
    else:
        return heatmap2keyptsBatch(heatmap)


def heatmap2keyptsSingle(heatmap):
    """
    Converts given heatmap into x and y coordinates of joints.

    Parameters
    ----------
    heatmap : np.array, shape = (Y, X, numJoints)
        Heatmap of different joints.

    Returns
    -------
    np.array, shape = (numJoints, 2)
        Array containing x and y coordinates for all joints.
    """
    # Flattening out X and Y dimensions, and finding the index of max value
    # for each joint. We will convert the indices into x and y cooordinates
    # using the resolution of the heatmap.
    heatmapWidth = heatmap.shape[1]
    numJoints = heatmap.shape[2]
    idx = np.argmax(heatmap.transpose([2, 0, 1]).reshape(numJoints, -1), axis=1)

    # Duplicating index and converting to column and row indices => x and y
    keyPts = idx.reshape(-1, 1).repeat(2, 1)
    keyPts[:, 0] = ((keyPts[:, 0] - 1) % heatmapWidth) + 1
    keyPts[:, 1] = np.floor((keyPts[:, 1] - 1) / heatmapWidth) + 1
    keyPts = np.int16(keyPts)

    return keyPts


def heatmap2keyptsBatch(heatmap):
    """
    Converts given batch of heatmaps into batch of x and y coordinates of joints.

    Parameters
    ----------
    heatmap : np.array, shape = (Batch, Y, X, numJoints)
        Heatmap of different joints.

    Returns
    -------
    np.array, shape = (Batch, numJoints, 2)
        Array containing x and y coordinates for all joints.
    """
    # Flattening out X and Y dimensions, and finding the index of max value
    # for each joint. We will convert the indices into x and y cooordinates
    # using the resolution of the heatmap.
    batchSize = heatmap.shape[0]
    heatmapWidth = heatmap.shape[2]
    numJoints = heatmap.shape[3]
    idx = np.argmax(heatmap.transpose([0, 3, 1, 2]).reshape(batchSize, numJoints, -1), axis=2)

    # Duplicating index and converting to column and row indices => x and y
    keyPts = idx.reshape(batchSize, -1, 1).repeat(2, 2)
    keyPts[..., 0] = ((keyPts[..., 0] - 1) % heatmapWidth) + 1
    keyPts[..., 1] = np.floor((keyPts[..., 1] - 1) / heatmapWidth) + 1
    keyPts = np.int16(keyPts)

    return keyPts


class HeatmapProcessor():
    """
    HeatmapProcessor provides a convenient way of coverting joint location
    co-ordinates into heatmaps and vice-versa.
    """

    def __init__(self, inHeight, inWidth, outHeight, outWidth, numJoints, scaleAware=False):
        """
        Initializes the HeatmapProcessor object with given options.

        Parameters
        ----------
        inHeight : int
            Height of image on which key points have been labelled.
        inWidth : int
            Width of image on which key points have been labelled.
        outHeight : int
            Output heatmap height
        outWidth : int
            Output heatmap width
        numJoints : int
            Number of joints = number of heatmap channels
        scaleAware: bool, optional
            The radius of the gaussian kernel placed at joints is
            computed as 1/64th size of the image width or height
            (which ever is largest). However is this is set to
            True, than the radius will be scaled up by the ratio
            of the size of image to the size of the bounding box
            encompassing all joints.  
        """
        self.inHeight = inHeight
        self.inWidth = inWidth
        self.outHeight = outHeight
        self.outWidth = outWidth
        self.numJoints = numJoints
        self.scaleAware = scaleAware

        self.needToRescaleWidth = inWidth != outWidth
        self.needToRescaleHeight = inHeight != outHeight

        self.imgSize = np.sqrt(outHeight * outWidth)
        self.sigma = np.round(self.imgSize / 64, decimals=1)
        self.gaussianKernelCache = {}

    def getGaussianKernel(self, sigma):
        """
        Returns a 2D gaussian kernel with the given radius.

        Parameters
        ----------
        sigma : float
            Radius of the gaussian kernel.

        Returns
        -------
        np.array
            2D gaussian kernel
        """
        if sigma in self.gaussianKernelCache:
            return self.gaussianKernelCache[sigma]

        # Creating a 2D gaussian kernel which we will drop at joint locations.
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.gaussianKernelCache[sigma] = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        return self.gaussianKernelCache[sigma]

    def rescaleKeyPts(self, keypts):
        # Normalize to 0.0 - 1.0 and rescale
        if self.needToRescaleWidth:
            keypts[:, 0] = (keypts[:, 0] / self.inWidth) * self.outWidth

        if self.needToRescaleHeight:
            keypts[:, 1] = (keypts[:, 1] / self.inHeight) * self.outHeight

        return keypts

    def keypts2heatmap(self, keypts, sigma=-1.0):
        """
        Converts given set of keypoints into a 3D heatmap with
        each joint along a different channel.

        Parameters
        ----------
        keypts : np.array, shape = (numJoints, 2)
            Array of key points, with the co-ordinates along the second dimension.
        sigma : float, optional
            Radius of gaussian kernel to use for producing heatmaps, by default -1.0
            which makes the method compute the radius based on the scale of the image
            or the bounding box (if scaleAware was set to True)

        Returns
        -------
        np.array, shape = (Y, X, numJoints)
            Heatmap
        """
        heatmap = np.zeros((self.outHeight, self.outWidth, self.numJoints), dtype=np.float32)
        keypts = self.rescaleKeyPts(keypts)

        if sigma < 0:
            if self.scaleAware:
                # TODO: (shahruk) re-visit how sigma should be scaled wrt to bbox size.
                xMin, xMax = np.min(keypts[:, 0]), np.max(keypts[:, 0])
                yMin, yMax = np.min(keypts[:, 1]), np.max(keypts[:, 1])
                bboxSize = np.sqrt((xMax - xMin) * (yMax - yMin))
                bboxToImgRatio = self.imgSize / bboxSize
                sigma = np.round(self.sigma * bboxToImgRatio, decimals=1)
            else:
                sigma = self.sigma

        g = self.getGaussianKernel(sigma)

        for idx, pt in enumerate(keypts):
            x, y = int(pt[0]), int(pt[1])

            # Checking if occlusion labels present.
            occluded = False
            if len(pt) > 2:
                occluded = pt[2]

            if occluded == 1:
                # print(f"skipping occluded key point, index={idx} coordinates=({x}, {y})")
                continue

            # Checking if key point coordinates are within image dimensions.
            if x < 0 or y < 0 or x >= self.outWidth or y >= self.outHeight:
                # print.debug(f"skipping key point out side heatmap region, index={idx} coordinates=({x}, {y})")
                continue

            # y = 80, x = 60
            # sigma = 2.5
            # ulx = 52
            # uly = 72
            # brx = 70
            # bry = 90
            ulx, uly = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            brx, bry = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            # a = 0
            # b = 18
            # c = 0
            # d = 18
            a, b = max(0, -uly), min(bry, self.outHeight) - uly
            c, d = max(0, -ulx), min(brx, self.outWidth) - ulx

            # aa = 72
            # bb = 90
            # cc = 52
            # dd = 70
            aa, bb = max(0, uly), min(bry, self.outHeight)
            cc, dd = max(0, ulx), min(brx, self.outWidth)

            # Setting gaussian map at location. Keeping maximum if overlap with
            # existing gaussian.
            heatmap[aa:bb, cc:dd, idx] = np.maximum(heatmap[aa:bb, cc:dd, idx], g[a:b, c:d])

        return heatmap

    def heatmap2keypts(self, heatmap):
        """
        Converts given heatmaps into x and y coordinates of joints.

        Parameters
        ----------
        heatmap : np.array, shape = (Y, X, numJoints)
            Heatmap of different joints.

        Returns
        -------
        np.array, shape = (numJoints, 2)
            Array containing x and y coordinates for all joints.
        """
        return heatmap2keypts(heatmap)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    hmapProcessor = HeatmapProcessor(160, 120, 40, 30, 14, scaleAware=True)

    testKeyPts = np.array([
        [44.85588508, 131.65848007, 0],
        [47.82159407, 104.37395737, 0],
        [53.15987025, 80.35171455, 0],
        [72.14040778, 81.24142725, 0],
        [76.88554217, 105.56024096, 0],
        [79.25810936, 131.95505097, 0],
        [35.95875811, 72.64087118, 0],
        [42.77988879, 53.95690454, 0],
        [51.67701576, 31.71408712, 0],
        [78.07182576, 33.19694161, 0],
        [83.70667285, 54.84661724, 0],
        [89.93466172, 72.93744208, 0],
        [64.13299351, 28.74837813, 0],
        [63.83642261, 10.06441149, 0],
    ])

    hmap = hmapProcessor.keypts2heatmap(testKeyPts, sigma=-1.0)
    keyPts = hmapProcessor.heatmap2keypts(hmap)
    print(keyPts.shape)
    print(f"Key Points recomputed from heatmap:\n{keyPts}\n")

    mse = np.mean(np.power(np.round(testKeyPts[:, 0:2]) - keyPts, 2))
    print(f"MSE between test key points and key points recomputed from heatmap = {mse}")

    hmapCombined = np.squeeze(np.sum(hmap, axis=-1))
    plt.matshow(hmapCombined)
    plt.show()

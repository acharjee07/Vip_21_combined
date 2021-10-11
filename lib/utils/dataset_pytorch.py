import os
import numpy as np

from lib.cover_augmentations import get_custom_aug

from lib.utils import HeatmapProcessor
import cv2
import torch

class SLPdataset:

    def __init__(self,
                 config,
                 imgPaths,
                 keyPts,
                 
                 normalizeImg=False,
                 normalizeKeyPts=False,
                 outputHeatmap=False,
                 heatmapRes=None,
                 heatmapSigmaRange=(3.0, 1.5),
                 leftRightJointPairIndexes=None,
                 probFlipH=0.0,
                 probMaskRandom=0.0,
                 probGaussianNoise=0.0,
                 probAttu=1,
                 shuffle=False,
                 numRepeatIter=1,
             
                 resize=False,
                epoch=(30,0)):
        
        self.resize=resize
        self.config=config
        
        self.imgPaths = imgPaths
        
        self.keyPts = keyPts
        self.leftRightJointPairIndexes = leftRightJointPairIndexes
        self.normalizeImg = normalizeImg
        self.normalizeKeyPts = normalizeKeyPts
        
        self.probFlipH = probFlipH
        self.probMaskRandom = probMaskRandom
        self.probGaussianNoise = probGaussianNoise
        self.probAttu=probAttu

        if self.probFlipH > 0.0 and self.leftRightJointPairIndexes is None:
            raise ValueError("need left-right joint pair indexes for horizontal flipping")

        self.numRepeatIter = numRepeatIter
        if self.numRepeatIter <= 0:
            self.numRepeatIter = 1

        self.shuffle = shuffle
        self.idx = -1

        imgFile = self.imgPaths[0]
        img = self._loadImg(imgFile)
        if self.keyPts is not None:
            keyPts = self.keyPts[imgFile]

        self.imgRes = img.shape[:2]
        if self.keyPts is not None:
            self.numJoints = len(keyPts)

        self.heatmapRes = heatmapRes
        if self.heatmapRes is None:
            self.heatmapRes = self.imgRes  # No downsampling

        sigmaStart, sigmaEnd = heatmapSigmaRange
        if epoch is not None:
            self.heatmapSigmaSchedule = np.arange(sigmaStart, sigmaEnd, -(sigmaStart-sigmaEnd)/epoch[0])
            self.heatmapSigmaIdx = 0
            if epoch[1]>=epoch[0]:
                self.heatmapSigma = self.heatmapSigmaSchedule[epoch[0]-1]
            else:

                self.heatmapSigma = self.heatmapSigmaSchedule[epoch[1]]
        else:
            self.heatmapSigma = 4
            
        
        self.outputHeatmap = outputHeatmap
        if self.keyPts is not None:
            self.heatmapProcessor = HeatmapProcessor(
                self.imgRes[0], self.imgRes[1],
                self.heatmapRes[0], self.heatmapRes[1],
                self.numJoints, scaleAware=True,
            )

        self.commonImagePath = os.path.commonpath(self.imgPaths)

    def __len__(self):
        return len(self.imgPaths) * self.numRepeatIter



    def dataShapes(self):
        sample = self._getSample()
        shapes = {}
        for name, x in sample.items():
            if hasattr(x, "shape"):
                shapes[name] = x.shape
            elif isinstance(x, str):
                shapes[name] = ()
            else:
                raise ValueError(f"unsupported type for dataloader '{type(x)}'")

        return shapes

    def getShortFilename(self, imgPath):
        return imgPath.replace(self.commonImagePath, "")

    def maskImageRandom(self, img, keyPts, imgW, imgH, centerX, centerY):
        maskW = int(imgW * np.random.uniform(0.50, 0.70))
        maskH = int(imgH * np.random.uniform(0.50, 0.70))
        centerXHalf = centerX / 2
        centerYHalf = centerY / 2
        maskX = np.random.randint(low=centerXHalf, high=imgW - maskW)
        maskY = np.random.randint(low=centerYHalf, high=imgH - maskH)

        img[maskY:maskY + maskH, maskX:maskX + maskW] *= np.random.uniform(0.1, 0.5)

        # Setting occlusion label for masked key points.
        maskedPts = np.logical_and(
            np.logical_and(  # Masked X Points
                np.greater_equal(keyPts[:, 0], maskX),
                np.less_equal(keyPts[:, 0], maskX + maskW),
            ),
            np.logical_and(  # Masked Y Points
                np.greater_equal(keyPts[:, 1], maskY),
                np.less_equal(keyPts[:, 1], maskY + maskH),
            ),
        )

        keyPts[maskedPts, 2] = 1

        return img, keyPts
    def get_attenuation_augmentation(self,img):
        img=get_custom_aug(img)
        return img
        

    def horizontalFlipImage(self, img, keyPts, imgW, imgH):
        img = np.fliplr(img)
        keyPts[:, 0] = imgW - keyPts[:, 0]

        # Need to swap co-ordinates of left and right joints.
        for (left, right) in self.leftRightJointPairIndexes.values():
            keyPts[[left, right]] = keyPts[[right, left]]

        return img, keyPts

    def addGaussianNoise(self, img):
        # Images are scaled to be between 0-255 representing 20 to 40 degrees
        # Celsius => 12.75 pixel intensity per celsius.
        noiseMean = 12.75
        noise = np.random.normal(noiseMean, noiseMean / 2, img.shape)
        return img + noise

    def zoomInOut(self, img, keyPts, imgW, imgH):
        # TODO:
        raise NotImplementedError

    def _loadImg(self, path, colorMode="rgb"):
        # img = tf.keras.preprocessing.image.load_img(path, color_mode=colorMode)
        # img = np.float32(tf.keras.preprocessing.image.img_to_array(img))
        img=cv2.imread(path).astype(np.float32)
        return img

    def __getitem__(self, idx):
        
        imgFile = self.imgPaths[idx]
        return_path=imgFile
        if self.keyPts is not None:
            keyPts = self.keyPts[imgFile].copy()-1
            
            

        img = self._loadImg(imgFile, "rgb")
        imgH, imgW = img.shape[:2]
        centerX, centerY = imgW / 2.0, imgH / 2.0
        
        # Attuniation augmentation
        if np.random.binomial(1, p=self.probAttu):
            img=self.get_attenuation_augmentation(img)

        # Masking different parts of image randomly.
        if np.random.binomial(1, p=self.probMaskRandom):
            img, keyPts = self.maskImageRandom(img, keyPts, imgW, imgH, centerX, centerY)
        

        # Randomly flipping images horizontally.
        if np.random.binomial(1, p=self.probFlipH):
            if self.keyPts is not None: 
                img, keyPts = self.horizontalFlipImage(img, keyPts, imgW, imgH)
            else:
                img = np.fliplr(img)

        # Randomly adding gaussian white noise.
        if np.random.binomial(1, p=self.probGaussianNoise):
            img = self.addGaussianNoise(img)

        # Leaving out occlusion label for now.
        if self.keyPts is not None:
            keyPts = keyPts[:, 0:2]

        # Generate heatmap of joints
        heatmap = None
        if self.keyPts is not None:
            if self.outputHeatmap:
                # Passing in copy of keypoints since the processor may rescale
                # co-ordinates to fit output heat map resolution.
                heatmap = self.heatmapProcessor.keypts2heatmap(keyPts.copy(), sigma=self.heatmapSigma)
        img_not_normalized=img.copy()
        # Normalize image and key point coordinates.
        if self.normalizeImg:
#             img /= 255.0
            maxpixel=np.max(img)
            mean0=self.config['mean'][0]*maxpixel
            mean1=self.config['mean'][1]*maxpixel
            mean2=self.config['mean'][2]*maxpixel
            std0=self.config['std'][0]*maxpixel
            std1=self.config['std'][1]*maxpixel
            std2=self.config['std'][2]*maxpixel
            img[:,:,0]=((img[:,:,0]-mean0)/std0)
            img[:,:,1]=((img[:,:,1]-mean1)/std1)
            img[:,:,2]=((img[:,:,2]-mean2)/std2)
        if self.resize:
            dim=self.config['input_size'][1:]
            
            img=cv2.resize(img,dim)
            img_not_normalized=cv2.resize(img_not_normalized,dim)
            
        if self.keyPts is not None:
            if self.normalizeKeyPts:
                keyPts[:, 0] = (keyPts[:, 0] - centerX) / centerX
                keyPts[:, 1] = (keyPts[:, 1] - centerY) / centerY

        # return {"img": torch.from_numpy(img.copy()),
        #         # "keypts": keyPts,
        #         "heatmap": torch.from_numpy(heatmap.copy()),
        #         # "img_path": self.getShortFilename(imgFile)
        #         }
        img=img.astype(np.float32)
        if self.keyPts is not None:
            heatmap=heatmap.astype(np.float32)
        img_not_normalized=img_not_normalized.astype(np.float32)

        img=torch.from_numpy(img.copy())
        img=img.permute(2,0,1)
        if self.keyPts is not None:
            heatmap=torch.from_numpy(heatmap.copy())
            heatmap=heatmap.permute(2,0,1)
        if self.keyPts is not None:
            return [img,
                    # "keypts": keyPts,
                    heatmap,
                    self.getShortFilename(imgFile),
                    img_not_normalized
                    ]
        else:
            return [img,
                    # "keypts": keyPts,
#                     heatmap,
#                     self.getShortFilename(imgFile),
                    img_not_normalized,
                    return_path
                    
                    ]
                    


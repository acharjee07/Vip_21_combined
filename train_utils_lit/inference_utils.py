



import cv2
import torch
from tqdm import tqdm

# from pylab import rcParams

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from bin.pytorch.train_utils import get_preds
from lib.metrics import calcAllPCKhBatch
from lib.data.dataset_pytorch import SLPdataset
from lib.datasets import SLPDatasetJointToLabels, SLPDatasetLeftRightJointPairs, loadImagePathsAndLabels

transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])

def get_loader(path,config,loader_type='slp',flip=True,):

    # train_path='./data/train/train/'
    # test_path='./data/test1/'
    if loader_type=='slp' :
        
        trainImgPathsall, trainKeyPts = loadImagePathsAndLabels(path, onlyAnnotated=False)
        trainImgPaths=trainImgPathsall[0:1350]
        unannotatedImgPaths=trainImgPathsall[1350:]
    if loader_type=='valid':
        validImgPaths, validKeyPts = loadImagePathsAndLabels(path, onlyAnnotated=False)

    if loader_type=='test':
        unannotatedImgPaths=loadImagePathsAndLabels(path, onlyAnnotated=False)[0]

    heatmapRes=(config['hm_size'][0],config['hm_size'][1])
    if loader_type=='slp'or loader_type=='test':
        if flip:
            Dataset = SLPdataset(config,unannotatedImgPaths, keyPts=None,leftRightJointPairIndexes=SLPDatasetLeftRightJointPairs,
                                   outputHeatmap=True, heatmapRes=heatmapRes,
                                   normalizeImg=True, normalizeKeyPts=True, shuffle=False,probFlipH=1,probAttu=0,resize=True)
        else:
            Dataset = SLPdataset(config,unannotatedImgPaths, keyPts=None,
                                   outputHeatmap=True, heatmapRes=heatmapRes,
                                   normalizeImg=True, normalizeKeyPts=True, shuffle=False,probFlipH=0,probAttu=0,resize=True)
        img_paths=unannotatedImgPaths
        loader=DataLoader(Dataset, batch_size=5, shuffle=False, pin_memory=False, drop_last=True, num_workers=0)
        
    else:
        if flip:
            Dataset = SLPdataset(config,validImgPaths, validKeyPts,leftRightJointPairIndexes=SLPDatasetLeftRightJointPairs,
                                       outputHeatmap=True, heatmapRes=heatmapRes,
                                       normalizeImg=True, normalizeKeyPts=True, shuffle=False,probAttu=0,resize=True,probFlipH=1)
        else:
            Dataset = SLPdataset(config,validImgPaths, validKeyPts,
                                       outputHeatmap=True, heatmapRes=heatmapRes,
                                       normalizeImg=True, normalizeKeyPts=True, shuffle=False,probAttu=0,resize=True,probFlipH=0)
        img_paths=validImgPaths
        loader=DataLoader(Dataset, batch_size=5, shuffle=False, pin_memory=False, drop_last=True, num_workers=3)
        
    return img_paths,loader




def get_combined_pred(batchn,batchf,model):
    img=batchn[0]
    gt=batchn[1]
    img_original=batchn[3]
#     print(img.shape,gt.shape)
    pred=model(img.cuda()).detach().cpu()
    ##preds with flipped data given flipped keypoints 
    fim=batchf[0]
    fgt=batchf[1]
    predsflipped=model(fim.cuda()).detach().cpu()
    predsfixed=flip_hm(predsflipped)
    avg_pred=(predsfixed+pred)/2
    
    return [pred,predsfixed] ,avg_pred,img_original,gt

def flip_img(img):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
    timg=transform(img)
    return timg

def flip_hm(hm):
    hm1=transform(hm)
    hm2=transform(hm)
    
    for (left, right) in SLPDatasetLeftRightJointPairs.values():
        hm1[:,left]=hm2[:,right]
        hm1[:,right]=hm2[:,left]

    return hm1

def visual(image,annotation):

    color = (0, 255, 0)
    colorf = (255, 0, 0)

    thickness = 2

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    annotation=annotation
    joints=annotation.astype(int)

    image=cv2.line(image,tuple(joints[0]),tuple(joints[1]),colorf, thickness)
    image=cv2.line(image,tuple(joints[1]),tuple(joints[2]),colorf, thickness)
    image=cv2.line(image,tuple(joints[3]),tuple(joints[4]),color, thickness)
    image=cv2.line(image,tuple(joints[4]),tuple(joints[5]),color, thickness)
    image=cv2.line(image,tuple(joints[6]),tuple(joints[7]),colorf, thickness)
    image=cv2.line(image,tuple(joints[7]),tuple(joints[8]),colorf, thickness)
    image=cv2.line(image,tuple(joints[9]),tuple(joints[10]),color, thickness)
    image=cv2.line(image,tuple(joints[10]),tuple(joints[11]),color, thickness)
    image=cv2.line(image,tuple(joints[9]),tuple(joints[12]),color, thickness)
    image=cv2.line(image,tuple(joints[8]),tuple(joints[12]),color, thickness)
    image=cv2.line(image,tuple(joints[2]),tuple(joints[3]),color, thickness)
    image=cv2.line(image,tuple(joints[12]),tuple(((joints[2]+joints[3])/2).astype(int)),color, thickness)
    image=cv2.line(image,tuple(joints[12]),tuple(joints[13]),color, thickness)
    return (image)



def get_combined_pred_unannotated(batchn,batchf,model):
    img=batchn[0]
    
    img_original=batchn[1]
#     print(img.shape,gt.shape)
    pred=model(img.cuda()).detach().cpu()
    ##preds with flipped data given flipped keypoints 
    fim=batchf[0]
    
    predsflipped=model(fim.cuda()).detach().cpu()
    predsfixed=flip_hm(predsflipped)
    avg_pred=(predsfixed+pred)/2
    
    return [pred,predsfixed] ,avg_pred,img_original



def show_predicitons(idx,preds,gt,imgs):
    preds_heatmaps,preds_cordinates=preds
    gt_heatmaps,gt_cordinates=gt
    plt.subplot(2, 2, 1)
    plt.title("Predicted")
#     print(preds_cordinates[x])
    plt.imshow(visual(imgs[x].numpy().astype(np.uint8),preds_cordinates[x].numpy()*4))


    plt.subplot(2, 2, 2)
    plt.title("Ground Truth")

    plt.imshow(visual(imgs[x].numpy().astype(np.uint8),gt_cordinates[x].numpy()*4))

    plt.subplot(2, 2, 3)
    plt.title("Predicted")
    hmp=preds_heatmaps[x].permute(1,2,0).numpy()
    hmp=np.squeeze(np.max(hmp, axis=-1))
    plt.imshow(hmp)


    plt.subplot(2, 2, 4)
    plt.title("Ground Truth")
    hmg=gt_heatmaps[x].permute(1,2,0).numpy()
    hmg=np.squeeze(np.max(hmg, axis=-1))
    plt.imshow(hmg)
    
def get_pck_single(gt,pred,th):
    head_size=np.linalg.norm(gt[12]-gt[13])
    distances=np.linalg.norm(gt-pred,axis=1)
    pck=distances>head_size*th
    return pck




def get_results(model,checkpoint,prediction_type,
                loadern,loaderf):

    validImgPaths, validKeyPts = loadImagePathsAndLabels('./data/train/train/', onlyAnnotated=False)
    trainImgPathsall, trainKeyPts = loadImagePathsAndLabels('./data/train/train/', onlyAnnotated=False)
    trainImgPaths=trainImgPathsall[0:1350]
    unannotatedImgPaths=trainImgPathsall[1350:]

        
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.cuda()
    model.eval()
    imgs=[]
    gt_heatmaps=[]
    preds_heatmaps=[]
    gt_cordinates=[]
    preds_cordinates=[]
    acc_all=[]
    acc_mean=[]
    
    for batchn,batchf in tqdm(zip(loadern,loaderf)):
        if prediction_type=='valid':
            individual_pred ,avg_pred, original_img,gt=get_combined_pred(batchn,batchf,model)
            accm,acca=calcAllPCKhBatch(get_preds(gt),get_preds(avg_pred),th=.5)
            imgs.extend(original_img)
            preds_cordinates.extend(get_preds(avg_pred))
            gt_cordinates.extend(get_preds(gt))
            preds_heatmaps.extend(avg_pred)
            gt_heatmaps.extend(gt)
            acc_all.append(acca)
            acc_mean.append(accm)
        else:
            _,avg_pred,original_img=get_combined_pred_unannotated(batchn,batchf,model)
            imgs.extend(original_img)
            preds_cordinates.extend(get_preds(avg_pred))
            preds_heatmaps.extend(avg_pred)
            
    if prediction_type=='valid':
        df=pd.DataFrame(acc_all)
        print(df.mean(axis = 0))
        print(np.mean(acc_mean))
        plt.imshow(visual(imgs[0].numpy().astype(np.uint8),preds_cordinates[0].numpy()*4))
        plt.show()
        return [
                [preds_heatmaps,preds_cordinates],
                
                [gt_heatmaps,gt_cordinates],
                imgs,
                df]
    else:
        return [preds_cordinates,imgs,unannotatedImgPaths, preds_heatmaps]
        
  



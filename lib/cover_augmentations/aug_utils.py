import pickle
from numpy.lib.stride_tricks import as_strided
import numpy as np
import random
import cv2
import os

current_dir=os.getcwd()
further_dir='{}/lib/cover_augmentations/aug_data'.format(current_dir)
def save_list(sample_list,name):
    
    file_name = name
    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()
def load_list(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

def getAttenuation(maxIR1,maxIR2,attenuation1,attenuation2,maxIR_Value):
    select_ind1=np.argmin(np.abs(maxIR1-maxIR_Value))
    select_ind2=np.argmin(np.abs(maxIR2-maxIR_Value))
    select_ind1=np.array(select_ind1)
    select_ind2=np.array(select_ind2)
    attenuation=attenuation2[select_ind2]/attenuation1[select_ind1]
    return attenuation

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


def getFold(att):
    Num_mask=random.randint(2,3)
    mask=np.array([[0]])
    for i in range(Num_mask):
        k=random.randint(1,120)
        # print(np.str(k))
        ff=cv2.imread('{}/Fold/'.format(further_dir)+np.str(k)+'.jpg',0)
        # ff=cv2.imread('lib/data/Fold/1.jpg')
        
        maskT=(ff<255)*1
        mask=mask+maskT
        mask[35:45]=1
    maskF=(mask>0)*att+(mask==0)
    return maskF




def getCovered(image,attenuation):
    lower_part=np.copy(image[35:150,20:100])
    #ave=np.sum(lower_part)/(lower_part.shape[1]*lower_part.shape[0])
    ave=pool2d(lower_part, kernel_size=40, stride=2, padding=0, pool_mode='avg')
    #ave2=pool2d(image, kernel_size=40, stride=2, padding=0, pool_mode='avg')
    mask=(lower_part>np.max(ave)*0.8)*1
    mask2=(image>np.max(ave)*0.8)*1
    # plt.imshow(mask2)
    lower_partE=mask*lower_part
    lower_partNE=(mask==0)*lower_part
    lower_partCE=attenuation*lower_partE
    newImage=np.copy(image)
    image_Covered=lower_partCE+lower_partNE
    image_Covered=cv2.GaussianBlur(image_Covered,(3,3),0)
    image_Covered=(image_Covered<30)*30+(image_Covered>=30)*image_Covered
    newImage[35:150,20:100]=image_Covered
    newImage=(mask2==0)*newImage*np.min(image_Covered*mask+(mask==0)*1000)*1.3/np.max((mask2==0)*newImage)+mask2*newImage
    att=np.random.uniform(0.7,0.8)
    att2=np.random.uniform(0.7,0.8)
    fold=getFold(att)
    mask3=np.copy(mask)
    mask3[0:10]=1*att
    newImage[35:150,20:100]=newImage[35:150,20:100]*(mask*fold[35:150,20:100])+newImage[35:150,20:100]*(mask==0)
    #newImage[45:150,20:100]=newImage[45:150,20:100]*(fold[45:150,20:100])+newImage[45:150,20:100]*(fold[45:150,20:100]==0)
    #newImage[35:45,20:100]=newImage[35:45,20:100]*att2
    newImage=cv2.GaussianBlur(newImage,(3,3),0)


    return newImage

# further_dir='{}/lib/cover_augmentation/aug_data'.format(current_dir)

maxIR1=load_list('{}/maxIR1'.format(further_dir))
attenuation1=load_list('{}/attenuation1'.format(further_dir))

maxIR2=load_list('{}/maxIR2'.format(further_dir))
attenuation2=load_list('{}/attenuation2'.format(further_dir))

maxIR3=load_list('{}/maxIR3'.format(further_dir))
attenuation3=load_list('{}/attenuation3'.format(further_dir))



def get_custom_aug(image):

#     image=cv2.imread(path,0)
    image=image[:,:,0]

    att2=getAttenuation(np.array(maxIR1),np.array(maxIR2),np.array(attenuation1),np.array(attenuation2),np.max(image))
    att3=getAttenuation(np.array(maxIR1),np.array(maxIR3),np.array(attenuation1),np.array(attenuation3),np.max(image))
    if att2<0.55:
        att2=0.65
    if att3<0.35:
        att2=0.4
    image2=getCovered(image,att2)
    image3=getCovered(image,att3)
    imlist=[image,image2,image3]
    rnd_im=random.choice(imlist)
    ch=np.dstack([rnd_im]*3)
    chi=ch.astype(np.float32)
    return chi

import torch
import matplotlib.pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from train_utils_lit import get_loader,get_results

from models import mxy
from train_utils_lit import LitPose
class Config:
    th=.5
    seed=42
    n_epoch=100
    #### schedular###
    lr = 1e-4
    max_lr = .9e-3
    pct_start = 0.3
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    ######
    betas=(0.9, 0.999)
    eps=1e-08
    weight_decay=0.01
    amsgrad=True
    steps_per_epoch=241
seed_everything(Config.seed)


data_config={
'input_size': (3, 440, 440),
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'crop_pct': 1.0,
 'hm_size':(112,112)

 }





model = mxy

lit_model = LitPose(
    plConfig=Config,
    data_config=data_config,
    model=model,
    phase=1
    )

valid_path='data/slp/valid/valid/'
_,flipped_loader_test = get_loader(
                                                path=valid_path,
                                                config=data_config,
                                                loader_type='valid',
                                                flip=True)
_, notflipped_loader_test=get_loader(
                                    path=valid_path,
                                    config=data_config,
                                    loader_type='valid',
                                    flip=False
                                    )
checkpoint='epoch=00-valid_auc=0.3336-valid_acc=6.6984-train_loss=0.1035-train_acc=4.5503.ckpt'
result=get_results(
                        model=lit_model,
                        checkpoint=checkpoint,
                        prediction_type='valid',
                        loadern=notflipped_loader_test,
                        loaderf=flipped_loader_test
                        )
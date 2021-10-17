import torch
import matplotlib.pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from models import mxy
from train_utils_lit import LitPose

import wandb
wandb.login(key='b19924dbbd8814abfc6253cb43cb4f741cdd4f98')  ##logging in sanjay
from pytorch_lightning.loggers import WandbLogger


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
'input_size': (3, 388, 388),
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'crop_pct': 1.0,
 'hm_size':(104,104),
 'batch_size':24

 }





model = mxy

lit_model = LitPose(
    plConfig=Config,
    data_config=data_config,
    model=model,
    phase=0
    )
logger= WandbLogger(name='',project='Mpii training')  

checkpoint_callback=ModelCheckpoint(monitor='valid_auc',
                                   save_top_k=1,
                                   save_last=True,
                                   save_weights_only=False,
                                   filename='{epoch:02d}-{valid_auc:.4f}-{valid_acc:.4f}-{train_loss:.4f}-{train_acc:.4f}',
                                    verbose=False,
                                    mode='max',
                                    dirpath='./Saved_weights/Mpii_pretrained'
                                   )




trainer = Trainer(auto_lr_find=Config.lr,
    max_epochs=Config.n_epoch,
    gpus=[0],
    callbacks=checkpoint_callback,
    logger=logger,

    weights_summary='top',
    amp_backend='native'
)

trainer.fit(lit_model)
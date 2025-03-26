#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os

import typing
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint,ModelSummary

from L_module import MicroCNN

class LearningRateLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Log the learning rate of the first param group
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        pl_module.log('learning_rate', lr, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

def main():

    global args

    print(args)

    seed_everything(42, workers=True)

    # Initialize model and dataloader classes
    model = MicroCNN(
                     data_path = args.data_path,
                     output_val=args.i_outputs,
                     batch_size=args.batch_size,
                     learning_rate=args.lr,
                     arcstr=args.architecture,
                     reg=args.regularization,
                     transform=True,
                     augment=True,
                     aug_factor=args.data_aug_factor,
                     val_split_frac=0.2,
                     test_split_frac=0,
                     split_by_job=~args.dont_keep_sets_together,
                     contrast=args.contrast
                    )

    lr_logger = LearningRateLogger()

    # Checkpoints the model for best validation loss
    best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="best_loss-{epoch:03d}-{loss:.6f}",
    )
    
    # Checkpoints the model every 2 epochs
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor="epoch",
        mode="max",
        every_n_epochs=args.n_epochs // 2,
        save_last=True,
        filename="loss-{epoch:03d}",
    )

    # Implement model training
    trainer = Trainer(
        default_root_dir=args.dir,
        max_epochs=args.n_epochs,
        # set this to auto when GPU available
        accelerator="auto",
        strategy=L.strategies.DDPStrategy(find_unused_parameters=False),
        deterministic=True,
        callbacks=[
            TQDMProgressBar(),
            best_callback,
            checkpoint_callback,
            lr_logger
        ],
        # Model weights and parameters are save in checkpoint.
        # Supply this if you want to start from previous traininge
        resume_from_checkpoint=args.ckpt,
    )

    trainer.fit(model)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data Saving Parameters
    parser.add_argument(
        "--dir", type=str, default="./resultsLR2HR", help="directory that saves all the logs"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="synth_CT_training_20_anode_norm.h5",
        help="file name where the data belongs",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="lightning checkpoint from where training can be restarted",
    )
    # Model training
    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.8,
        help="scheduler factor to reduce every 10 epochs",
    )
    parser.add_argument(
        "--i_outputs",
        type=str,
        default='(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)',
        help="Which outputs to train on? Enter with commas and no spaces. Key: 0,1,2 = f1,2,3.  3,4,5 = tort1,2,3. 6,7,8 = davg1,2,3.  9=tpb.  10,11,12 = sa12,sa13,sa23.  13,14,15 = std1,2,3",
    )
    
    parser.add_argument(
        "-d",
        "--data_aug_factor",
        type=float,
        default=4,
        help="Data augmentation factor.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        default="c16,a1k2s2,c32,a1k2s2,c64,a1k2s2,c128,a1k2s2,d128,d64",
        help="CNN architecture. [(c)onv3d/(d)ense/(a)vgpool/(m)axpool][# filters or neurons, int][k3, kernel size][s1, stride][pv, padding - s(a)me or (v)alid]. Thusly for each layer separated by commas. See docstring at top of this script file for more details.",
    )
    parser.add_argument(
        "--validation_split", type=float, default=0.2, help="Validation split (0 to 1)."
    )
    parser.add_argument(
        "--test_split", type=float, default=0, help="Test split (0 to 1)."
    )
    parser.add_argument(
        "--dont_keep_sets_together",
        action="store_true",
        default=False,
        help="Suppress the default behavior of automatically keeping families of microstructures together across the test-train split.",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        choices=["l1", "l2", "l1_l2", "None"],
        default="None",
        help="Regularization type.",
    )
    parser.add_argument(
        "--dropout_frac",
        type=float,
        default=0,
        help="Dropout fraction for Conv3D layers (default 0, no droupout).",
    )
    parser.add_argument(
        "--transfer_learning_epoch_factor",
        type=float,
        default=0,
        help="For using transfer learning (beginning training for parameter N from the final weights of parameter 1)... for example if N_epochs = 100, parameter 1 will train for 100. Then for parameter 2, if transfer_learning_epoch_factor=0.4, parameter 2 will train for 40 epochs (they will be labelled as epochs 101 - 140). If you want transfer learning to be OFF, set this to 0! This will cause them all to retrain from the beginning (default)",
    )
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=1.0,
        help="clipnorm for Adam optimizer (default 1)",
    )

    parser.add_argument(
        "--contrast",
        type=str,
        default='A',
        choices=['A','C'],
        help="X-ray contrast used for data. Important for data normalization.",
    )
    
    args = parser.parse_args()
    
    try:
        args.i_outputs=list(eval(args.i_outputs))
    except:
        args.i_outputs=[eval(args.i_outputs)]
        
    main()






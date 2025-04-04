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

from data import *
from architecture import *

class MicroCNN(pl.LightningModule):
    def __init__(self, 
                 data_path, 
                 output_val=[0,1,2], 
                 batch_size=32, 
                 learning_rate=1e-3, 
                 arcstr="C64k3s2pa,D256",
                 reg = 'l1',
                 transform = True,
                 augment=True,
                 aug_factor=4,
                 val_split_frac = 0.2,
                 test_split_frac = 0,
                 split_by_job = False,
                 contrast='C'
                ):

        
        super().__init__()
        arcstr = arcstr.lower()

        
        self.data_path = data_path
        self.output_val = output_val
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.arcstr = arcstr
        self.transform = transform
        self.augment = augment
        self.aug_factor = aug_factor
        self.val_split_frac = val_split_frac
        self.test_split_frac = test_split_frac
        self.split_by_job = split_by_job
        self.contrast=contrast

        with h5py.File(data_path, "r") as f:
            X = f["X"][0]  
          

        
        
        # Initialize the model using the CNNModel class
        self.model = CNNModel(img_shape=(1,X.shape[0], X.shape[1], X.shape[2]), 
                              out_shape= len(list(output_val)),
                              arcstr=self.arcstr,
                              reg = reg,)
        
        # Auto-log all the hyperparameters to your logger
        self.save_hyperparameters()

    def forward(self, x):
        # Forward pass
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        
        y_hat = self(x)
        loss = F.mse_loss(y_hat.flatten(), y.flatten())
        self.log('train_loss', loss,on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step to monitor performance
        x, y = batch
        y_hat = self(x)
        
        val_loss = F.mse_loss(y_hat.flatten(), y.flatten())
        self.log('val_loss', val_loss,on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        # Configure the optimizer and optionally a scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage=None):
        # Split the dataset for training and validation
        
        if not self.split_by_job:
            full_dataset = Microstructures(self.data_path, self.output_val,self.transform,self.augment,self.aug_factor,contrast=self.contrast)
            val_size = int(self.split_frac * len(full_dataset))
            train_size = len(full_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        else:
            full_dataset = Microstructures(self.data_path, self.output_val,self.transform,self.augment,self.aug_factor,contrast=self.contrast,job_group=True)
            self.train_dataset, self.val_dataset = full_dataset.split_by_job_group(full_dataset,self.val_split_frac)

    def train_dataloader(self):
        # Training data loader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4,shuffle=True)

    def val_dataloader(self):
        # Validation data loader
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=4)

    
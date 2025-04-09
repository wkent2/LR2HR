import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as L
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from architecture import *
import pandas as pd
from L_module import MicroCNN
import argparse
import yaml
from data import Microstructures
from tqdm import tqdm

def parseargs():
    p = argparse.ArgumentParser(description="Plots results from CNN training")
    p.add_argument('version',type=int,help="training version")
    p.add_argument('-r',type=str,default="./resultsLR2HR",help="folder containing all training results")
    p.add_argument('-b',type=bool,default=True,help="Whether to plot the best or the last checkpoint")
    p.add_argument('-data',type=str,default=None,help="Option to plot different data")
    p.add_argument('-mi',type=str,default=None,help="Option to plot different microstructure characteristic data")
    p.add_argument('-dmap',type=str,default=None,help="Option to plot with dmap values")
    p.add_argument('-s',type=int,default=42,help="Random seed")
    p.add_argument('-c',type=str,default=None,help="Option to change contrast")

    args = p.parse_args()
    
    return args

def save_names(train_dataset,val_dataset,path_to_res):

    train_path = os.path.join(path_to_res,'train_names_plot.txt')
    val_path = os.path.join(path_to_res,'val_names_plot.txt')

    with open(train_path, 'w') as f:
        f.write(f"# {len(train_dataset.names)} samples\n")
        for name in train_dataset.names:
            f.write(f"{name}\n")

    with open(val_path, 'w') as f:
        f.write(f"# {len(val_dataset.names)} samples\n")
        for name in val_dataset.names:
            f.write(f"{name}\n")

def load_hyperparameters(file_path):
    with open(file_path, "r") as file:
        hyperparams = yaml.safe_load(file)  # Load YAML content safely
    return hyperparams

def check_file_exists(filepath,err_msg):
    if not os.path.exists(filepath):
        print(err_msg)
        exit()


def get_checkpoint_path(results_path,best):

    # Construct checkpoints folder path
    ckpt_folder = os.path.join(results_path,'checkpoints')

    # Check that checkpoints folder exists
    check_file_exists(ckpt_folder,"There is no checkpoints folder in the version folder")

    # Checks if using best checkpoint
    if best:
        # Checks the best checkpoint exists
        files = [file for file in os.listdir(ckpt_folder) if file.endswith('.ckpt')]
        checkp = next((f for f in files if f.startswith('best_loss')), None)
        checkp = os.path.join(ckpt_folder,checkp)
        check_file_exists(checkp,"Cannot find the best_loss-.....ckpt checkpoint")
    else:
        # Check that last checkpoint exists
        checkp = os.path.join(ckpt_folder,'last.ckpt')
        check_file_exists(checkp,"Cannot find the last.ckpt checkpoint")

    return checkp




if __name__ == "__main__":

    # Parse command line arguments
    args = parseargs()

    # Find .yaml file with hyperparameters
    version_str = 'version_'+str(args.version)
    path_to_res = os.path.join(args.r,'lightning_logs',version_str)
    check_file_exists(path_to_res,"Cannot find version folder")

    # Checks if hparams.yaml exists along path
    check_file_exists(os.path.join(path_to_res,'hparams.yaml'),f"Error: There is no hparams.yaml file in the version folder")
    # Get hyperparameters 
    hparams = load_hyperparameters(os.path.join(path_to_res,'hparams.yaml'))

    # Applies random seed to everything
    seed_everything(args.s, workers=True)

    print("Constructing dataset")

    if args.data:
        hparams['data_path'] = args.data

    if args.c:
        hparams['contrast'] = args.c

    print("Constructing dataset from",hparams['data_path'], "with",hparams['contrast'],"contrast")

    # Load data
    full_dataset = Microstructures(hparams['data_path'], 
                                    hparams['output_val'],
                                    transform=hparams['transform'],
                                    augment=hparams['augment'],
                                    factor=hparams['aug_factor'],
                                    remove_bad=True,
                                    contrast=hparams['contrast'],
                                    job_group=~hparams['split_by_job'],
                                    seed=args.s)


    # Applies extra random seed. Not sure why it needs this but data does not 
    # split correctly without it. 
    generator1 = torch.Generator().manual_seed(args.s)

    if ~hparams['split_by_job']:
        val_size = int(hparams['val_split_frac'] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        save_names(train_dataset,val_dataset,path_to_res)
    else:
        train_dataset, val_dataset = full_dataset.split_by_job_group(full_dataset,hparams['val_split_frac'],seed=args.s)
        save_names(train_dataset,val_dataset,path_to_res)
        print("Microstructure were split by job group \n")
    
    # Initializes train and validation data loaders
    train_DL = DataLoader(train_dataset, batch_size=hparams['batch_size'], num_workers=1,shuffle=True)
    val_DL = DataLoader(val_dataset, batch_size=hparams['batch_size'],num_workers=1)

    # Get path to checkpoint
    chk_path = get_checkpoint_path(path_to_res,args.b)

    # Initialize model and dataloader classes
    model = MicroCNN.load_from_checkpoint(chk_path)

    # Figures out whether or not to use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Prints out model for sanity check
    for name, layer in model.named_children():
        print(f"Layer name: {name}, Layer: {layer}")

    # Sets model to evaluation mode (no gradient computations)
    model.eval()

    # Initializes arrays for storing data. 
    ys_train = []
    yhats_train = []
    ys_val = []
    yhats_val = []

    # Iterates through TRAINING data    
    for batch, (X,y) in enumerate(tqdm(train_DL)):
        y_hat = model(X.to(device))
        ys_train.append(y.detach().cpu().numpy())
        yhats_train.append(y_hat.cpu().detach().numpy())

    # Iterates through VALIDATION data
    for batch, (X,y) in enumerate(tqdm(val_DL)):
        y_hat = model(X.to(device))
        ys_val.append(y.detach().cpu().numpy())
        yhats_val.append(y_hat.detach().cpu().numpy())


    # Concatenates to numpy array 
    ys_train_full = np.concatenate(ys_train)
    yhats_train_full = np.concatenate(yhats_train)
    ys_val_full = np.concatenate(ys_val)
    yhats_val_full = np.concatenate(yhats_val)

    # Saves raw targets and predictions
    np.save(os.path.join(path_to_res,'ys_train.npy'),ys_train_full)
    np.save(os.path.join(path_to_res,'yhats_train.npy'),yhats_train_full)
    np.save(os.path.join(path_to_res,'ys_val.npy'),ys_val_full)
    np.save(os.path.join(path_to_res,'yhats_val.npy'),yhats_val_full)

    # PLOTS RESULTS 
    # This is just to quickly asses if the plotting was done 
    # correctly on Joule. Not final results 

    # Gets keys for the variables
    
    names = ['f1','f2','f3','tort1','tort2','tort3','davg1','davg2','davg3','tpb','tpbconnfrac','tpbtot','sa12','sa13','sa23']

    print("Plotting results")
    for i in range(len(names)):
        plt.close('all')
        plt.figure(figsize=(10,5))

        plt.suptitle(names[i],fontsize=16)

        fulldata = np.concatenate([ys_train_full,yhats_train_full,
                                    ys_val_full,yhats_val_full])

        bin_w = np.linspace(np.amin(fulldata),np.amax(fulldata),200)

        plt.subplot(1,2,1)
        plt.hist2d(ys_train_full[:,i],yhats_train_full[:,i],bins=bin_w,cmap='Blues',density=True)
        plt.plot(bin_w,bin_w,color='black',linewidth=0.5)
        plt.gca().set_box_aspect(1.0)
        plt.xlabel('Target',fontsize=12)
        plt.ylabel('Predicted',fontsize=12)
        plt.title('Training Data',fontsize=14)

        plt.subplot(1,2,2)
        plt.hist2d(ys_val_full[:,i],yhats_val_full[:,i],bins=bin_w,cmap='Blues',density=True)
        plt.plot(bin_w,bin_w,color='black',linewidth=0.5)
        plt.gca().set_box_aspect(1.0)
        plt.xlabel('Target',fontsize=12)
        plt.ylabel('Predicted',fontsize=12)
        plt.title('Validation Data',fontsize=14)


        savename = names[i]+'.png'
        plt.savefig(os.path.join(path_to_res,savename),dpi=300)







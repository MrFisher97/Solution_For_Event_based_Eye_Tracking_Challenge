"""
This script implements a PyTorch deep learning training pipeline for an eye tracking application.
It includes a main function to pass in arguments, train and validation functions.
The script also supports fine-grained deep learning hyperparameter tuning using YAML configuration files.

Author: Zengyu Wan
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wanzengy@mail.ustc.edu.cn
"""

import argparse, yaml, os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import MC_Loss as Loss

from dataset.ThreeET_plus import ThreeETplus_Eyetracking
import dataset.custom_transforms as T
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset

import importlib
import logging
import time

def log_setting(config):
    logger = logging.getLogger()
    ch = logging.StreamHandler() # Adding Terminal Logger

    # Adding File Logger
    log_dir = os.path.join(config['log_dir'], config['model'], config['timestamp'])
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir

    fh = logging.FileHandler(filename=os.path.join(log_dir, 'logger.txt'))
    fh.setFormatter(logging.Formatter("%(asctime)s  : %(message)s", "%b%d-%H:%M"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.setLevel(logging.INFO)
    logger.info(yaml.dump(config, sort_keys=False, default_flow_style=False))
    return logger

def train(model, train_loader, val_loader, criterion, optimizer, args, logger):
    best_val_loss = float("inf")
    mertric_record = None
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        model, train_loss, metrics = train_epoch(model, train_loader, criterion, optimizer, args)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f} \t ela: {metrics['ela']:.5}")
    
        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0 or epoch == 0:
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, args)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mertric_record = val_metrics
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.log_dir, \
                            f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))
                top_k_checkpoints(args)
                
            info = f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}\n"
            for k, v in val_metrics['val_p_acc_all'].items():
                info +=(f"\t {k[4:]}: {v:.4f} \n")
            info += f"\t p_error_all:{val_metrics['val_p_error_all']['val_p_error_all']}"
            logger.info(info)

    info = f"Best epoch {best_epoch}: Val loss: {best_val_loss:.4f} \n"
    for k, v in mertric_record['val_p_acc_all'].items():
        info += f"\t {k[4:]}: {v} \n"
    info += f"\t {mertric_record['val_p_error_all']['val_p_error_all']} "
    logger.info(info)

    return model

def main(args):
    # Load hyperparameters from YAML configuration file
    with open('configs/' + args.config_file) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        if args.timestamp is None:
            config['timestamp'] = time.strftime('%y%m%d%H%M%S', time.localtime(time.time())) # Add timestamp with format mouth-day-hour-minute

    logger = log_setting(config)
    args = argparse.Namespace(**config)

    # Define your model, optimizer, and criterion
    model = importlib.import_module(f"model.{args.model}").Model(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    criterion = Loss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), 
                                        reduction='mean')

    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

    # First we define the label transformations
    label_transform = transforms.Compose([
        T.ScaleLabel(factor),
        T.TemporalSubsample(temp_subsample_factor),
        T.NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir,
                                              split="train",
                                              transform=transforms.Downsample(spatial_factor=factor), 
                                              target_transform=label_transform)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, 
                                            split="val", 
                                            transform=transforms.Downsample(spatial_factor=factor), 
                                            target_transform=label_transform)

    # Then we slice the event recordings into sub-sequences. 
    # The time-window is determined by the sequence length (train_length, val_length) 
    # and the temporal subsample factor.
    train_slicing_time_window = args.train_length * int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

    train_slicer = T.SliceByTimeEventsTargets(train_slicing_time_window, 
                                              overlap=train_slicing_time_window-train_stride_time, 
                                              seq_length=args.train_length, 
                                              seq_stride=args.train_stride, 
                                              include_incomplete=False)
    
    # the validation set is sliced to non-overlapping sequences
    val_slicing_time_window = args.val_length * int(10000/temp_subsample_factor) #microseconds
    val_slicer = T.SliceByTimeEventsTargets(val_slicing_time_window, 
                                            overlap=0, 
                                            seq_length=args.val_length, 
                                            seq_stride=args.val_stride, 
                                            include_incomplete=False)

    # After slicing the raw event recordings into sub-sequences, 
    # we make each subsequences into appropriate event representation.
    post_slicer_transform = transforms.Compose([
        T.SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), 
                                 overlap=0, 
                                 include_incomplete=True),
        T.EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2),
                           n_time_bins=args.n_time_bins, 
                           per_channel_normalize=args.voxel_grid_ch_normaization,
                           map_type='binary')
    ])

    # We use the Tonic SlicedDataset class to handle the collation of the sub-sequences into batches.
    train_data = SlicedDataset(train_data_orig, 
                               train_slicer, 
                               transform=post_slicer_transform, 
                               metadata_path=f"{args.metadata_dir}/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
    val_data = SlicedDataset(val_data_orig, 
                             val_slicer, 
                             transform=post_slicer_transform, 
                             metadata_path=f"{args.metadata_dir}/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

    # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
    train_data = DiskCachedDataset(train_data, 
                                   cache_path=f"{args.cache_dir}/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}",
                                   transforms=T.Jitter())
    val_data = DiskCachedDataset(val_data, 
                                 cache_path=f"{args.cache_dir}/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}",
                                 transforms=None)
 
    # Finally we wrap the dataset with pytorch dataloader
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=4, 
                              pin_memory=True)
    val_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=4)

    # Train your model
    model = train(model, train_loader, val_loader, criterion, optimizer, args, logger)

    # Save your model for the last epoch
    torch.save(model.state_dict(), os.path.join(args.log_dir, f"model_last_epoch{args.num_epochs}.pth"))

    from test import test
    test(args, model)

def init_seed(seed=1):
    import random
    import numpy as np
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    
    # a config file 
    parser.add_argument("--config_file", 
                        default="task.yaml", 
                        help="path to YAML configuration file")

    parser.add_argument("--timestamp",
                        default=None,
                        help="location of the cache version of the formatted dataset",)
    
    args = parser.parse_args()

    init_seed(2023)
    main(args)

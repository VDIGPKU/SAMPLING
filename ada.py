import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
import yaml
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize


logging = True
import yaml
yaml_path= "./configs/xxx.yaml"
flag = 0
def is_rank_zero(config1):
    return config1.rank == 0


import matplotlib

def read_yaml_all():
    try:

        with open(yaml_path,"r",encoding="utf-8") as f:
            data=yaml.unsafe_load(f)
            return data
    except:
        return None

def update_yaml(old_data,k,v):
    old_data[k]=v
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(old_data,f)

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  
    else:
  
        value = value * 0.
 
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  

    img = value[:, :, :3]

    return img


def log_images(img, depth, pred, config1, step):
    depth = colorize(depth, vmin=config1["min_depth"], vmax=config1["max_depth"])
    pred = colorize(pred, vmin=config1["min_depth"], vmax=config1["max_depth"])
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)


def main_worker(gpu, ngpus_per_node, config1,img,depth,train_data_loader, val_data_loader):
    config1["gpu"] = gpu
    if config1["flag"] :

        modelada = models.UnetAdaptiveBins.build(n_bins=config1["n_bins"], min_val=config1["min_depth"], max_val=config1["max_depth"],
                                              norm=config1["norm"])

  
        if config1["gpu"] is not None: 
            torch.cuda.set_device(config1["gpu"])
            modelada = modelada.cuda(config1["gpu"])

        config1["multigpu"] = False
        if config1["distributed"]:

            update_yaml(config1,"multigpu", True)
            update_yaml(config1, "gpu", 0)
            update_yaml(config1,"rank", config1["rank"] * ngpus_per_node + config1["gpu"])

            update_yaml(config1,"batch_size", int(config1["bs"] / ngpus_per_node) )

            update_yaml(config1,"workers", int((config1["num_workers"] + ngpus_per_node - 1) / ngpus_per_node) )
            print(config1["gpu"], config1["rank"], config1["bs"], config1["workers"])
            torch.cuda.set_device(config1["gpu"])
            modelada = nn.SyncBatchNorm.convert_sync_batchnorm(modelada)
            modelada = modelada.cuda(config1["gpu"])
            modelada = torch.nn.parallel.DistributedDataParallel(modelada, device_ids=[config1["gpu"]], output_device=config1["gpu"],
                                                              find_unused_parameters=True)

        elif config1["gpu"] is None:
            update_yaml(config1,"multigpu" ,True)
            modelada = modelada.cuda()
            modelada = torch.nn.DataParallel(modelada)
        update_yaml(config1, "flag", False)
    update_yaml(config1,"epoch", 0)
    update_yaml(config1,"last_epoch",-1)
    disparity_array,lossada = trainada(modelada, config1, img=img, depth=depth,train_data_loader=train_data_loader, test_data_loader=val_data_loader,epochs=config1["epochs"], lr=config1["lr"], device=config1["gpu"], root=config1["root"],
          experiment_name=config1["name"], optimizer_state_dict=None)
    return disparity_array,lossada



def trainada(modelada, config1, img, depth,train_data_loader, test_data_loader, epochs=10, experiment_name="Samp", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{4 }-tep{epochs}-lr{lr}-wd{0.1}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not config1["distributed"]) or config1["rank"] == 0)
    should_log = should_write and logging
    if should_log:
        tags = config1["tags"].split(',') if config1["tags"] != '' else None

    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if config1["chamfer"] else None

    modelada.train()

    if config1["same_lr"]:
        print("Using same LR")
        params = modelada.parameters()
    else:
        print("Using diff LR")
        m = modelada.module if torch.cuda.device_count()>1 else modelada
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=config1["wd"], lr=config1["lr"])
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    iters = len(train_data_loader)
    step = config1["epoch"] * iters
    best_loss = np.inf

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_data_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                              div_factor=config1["div_factor"],
                                              final_div_factor=config1["final_div_factor"])
    if config1["resume"] != '' and scheduler is not None:
        scheduler.step(config1["epoch"] + 1)
        optimizer.zero_grad()

    img = img.to(device)
    depth = depth.to(device)

    bin_edges, pred = modelada(img)

    mask = depth > config1["min_depth"]
    l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

    if config1["w_chamfer"] > 0:
        l_chamfer = criterion_bins(bin_edges, depth.float())
    else:
        l_chamfer = torch.Tensor([0]).to(img.device)

    loss = l_dense + config1["w_chamfer"] * l_chamfer

    step += 1
    scheduler.step()


    modelada.train()
    return bin_edges,loss
 

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def adamain(batch_size, num_bins, start, end,img,depth,train_data_loader,val_data_loader,device):
    
    with open('./configs/xxx.yaml', "r",encoding="utf-8") as f:
        config1 = yaml.unsafe_load(f)
    update_yaml(config1,"bs", batch_size)
    update_yaml(config1,"n_bins", num_bins)
    update_yaml(config1,"gpu",device)
    update_yaml(config1,"num_threads",11)
    update_yaml(config1,"mode", 'train')
    update_yaml(config1,"max_depth", start)
    update_yaml(config1,"min_depth", end)
    update_yaml(config1,"max_depth_eval",start)
    update_yaml(config1,"min_depth_eval",end)
    update_yaml(config1,"chamfer",config1["w_chamfer"] > 0)

    disparity_array,lossada = main_worker(device, 1, config1,img,depth,train_data_loader=train_data_loader, val_data_loader=val_data_loader)
    return disparity_array,lossada
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import random
import numpy as np
import shutil
import os
import pprint
import torch

import must.models.losses as losses
import must.models.optimizer as optim
import must.utils.checkpoint as cu
import must.utils.distributed as du
import must.utils.logging as logging
import must.utils.misc as misc

from must.datasets import loader
from must.models import build_model
from must.utils.meters import EpochTimer, SurgeryMeter, SurgeryMeterChunks
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import torch.nn as nn

logger = logging.get_logger(__name__)

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py

    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    tasks = cfg.TASKS.TASKS
    loss_funs = cfg.TASKS.LOSS_FUNC

    loss_dict = {task:losses.get_loss_func(loss_funs[t_id])(reduction=cfg.SOLVER.REDUCTION) for t_id,task in enumerate(tasks)}
    type_dict = {task:losses.get_loss_type(loss_funs[t_id],cfg.MODEL.PRECISION) for t_id,task in enumerate(tasks)}
    loss_weights = cfg.TASKS.LOSS_WEIGHTS
    for cur_iter, (inputs, labels, data, image_names) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            for idx, input in enumerate(inputs[0]):
                inputs[0][idx] = inputs[0][idx].cuda(non_blocking=True)
            if cfg.MODEL.PRECISION == 64:
                inputs[0] = inputs[0].double()

            for key, val in data.items():
                data[key] = val.cuda(non_blocking=True)
                if cfg.MODEL.PRECISION == 64:
                    data[key]  = data[key].double()

            for key, val in labels.items():
                labels[key] = val.cuda(non_blocking=True)
                if cfg.MODEL.PRECISION == 64:
                    labels[key]  = labels[key].double()
            
            if cfg.NUM_GPUS>1:
                image_names = image_names.cuda(non_blocking=True)
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            sequence_mask = data["sequence_mask"] if cfg.TEMPORAL_MODULE.CHUNKS else None
            if sequence_mask is not None:
                preds = model(inputs, sequence_mask)
            else:
                preds = model(inputs)
            # Explicitly declare reduction to mean and compute the loss for each task.
            loss = []
            for task in loss_dict:
                loss_fun = loss_dict[task]
                target_type = type_dict[task]

                if cfg.TEMPORAL_MODULE.CHUNKS == True:
                    bs, _, num_classes = preds[task].shape

                    preds[task] = preds[task].reshape(preds[task].shape[0] * preds[task].shape[1], -1)
                    sequence_mask = sequence_mask.reshape(-1)
                    
                    preds[task] = preds[task][~sequence_mask]
                    labels[task] = labels[task][~sequence_mask]

                loss.append(loss_fun(preds[task], labels[task].to(target_type))) 

        if len(loss_dict) >1:
            final_loss = losses.compute_weighted_loss(loss, loss_weights)
        else:
            final_loss = loss[0]
            
        # check Nan Loss.
        misc.check_nan_losses(final_loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.NUM_GPUS > 1:
            final_loss = du.all_reduce([final_loss])[0]
        final_loss = final_loss.item()

        # Update and log stats.
        train_meter.update_stats(None, None, final_loss, loss, lr)
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}

    for cur_iter, (inputs, labels, data, image_names) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            for idx, input in enumerate(inputs[0]):
                inputs[0][idx] = inputs[0][idx].cuda(non_blocking=True)
            if cfg.MODEL.PRECISION == 64:
                inputs[0] = inputs[0].double()

            for key, val in data.items():
                data[key] = val.cuda(non_blocking=True)
                if cfg.MODEL.PRECISION == 64:
                    data[key]  = data[key].double()

            for key, val in labels.items():
                labels[key] = val.cuda(non_blocking=True)
                if cfg.MODEL.PRECISION == 64:
                    labels[key]  = labels[key].double()
            
            if cfg.NUM_GPUS>1:
                image_names = image_names.cuda(non_blocking=True)
                    
        val_meter.data_toc()

        sequence_mask = data["sequence_mask"] if cfg.TEMPORAL_MODULE.CHUNKS else None

        # If calculation of features from the MTFE is enabled
        if cfg.MVIT_FEATS.ENABLE:
            preds = model(inputs, image_names)
    
        else:
            if sequence_mask is not None:
                preds = model(inputs, sequence_mask)
            else:
                preds = model(inputs)
        
        if cfg.NUM_GPUS:
            preds = {task: preds[task].cpu() for task in complete_tasks}

            if cfg.NUM_GPUS>1:
                image_names = image_names.cpu()
                image_names = torch.cat(du.all_gather_unaligned(image_names),dim=0).tolist()

                for task in preds:
                    preds[task] = torch.cat(du.all_gather_unaligned(preds[task]), dim=0)

        val_meter.iter_toc()

        for task in complete_tasks:
            if task not in region_tasks:
                preds[task] = preds[task].tolist()
        
        # Update and log stats.
        val_meter.update_stats(preds, image_names)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    if cfg.NUM_GPUS > 1:
        if du.is_master_proc():
            task_map, mean_map, out_files = val_meter.log_epoch_stats(cur_epoch)
        else:
            task_map, mean_map, out_files =  [0, 0, 0]
        torch.distributed.barrier()
    else:
        task_map, mean_map, out_files = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    return task_map, mean_map, out_files


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.MODEL.PRECISION == 64:
        model = model.double()

    # Calculating model info (param & flops). 
    # Remove if it is not working
    try:
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=True)
            for task in cfg.TASKS.TASKS:
                head = getattr(model, "extra_heads_{}".format(task))
                misc.log_model_info(head, cfg, use_train_input=False)
    except:
        pass

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
            
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    if cfg.TEMPORAL_MODULE.CHUNKS == False:
        # Create meters.
        train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
        val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")
    else:
        # Create meters.
        train_meter = SurgeryMeterChunks(len(train_loader), cfg, mode="train")
        val_meter = SurgeryMeterChunks(len(val_loader), cfg, mode="val") 

    # Perform final test
    if cfg.TEST.ENABLE:
        logger.info("Evaluating epoch: {}".format(start_epoch + 1))
        map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, start_epoch, cfg)
        if not cfg.TRAIN.ENABLE:
            return
    elif cfg.TRAIN.ENABLE:
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_map = {task: 0 for task in complete_tasks}
    best_mean_map = 0
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        
        if cfg.SOLVER.EARLY_STOPPING:
            assert cur_epoch != cfg.SOLVER.EARLY_STOPPING , "Early stopping"

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        if not cfg.MODEL.KEEP_ALL_CHECKPOINTS:
            del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
            if os.path.exists(del_fil):
                os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            map_task, mean_map, out_files = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                # Save best results
                if mean_map > best_mean_map:
                    best_mean_map = mean_map
                    logger.info("Best mean map at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in complete_tasks:
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in complete_tasks:
                    if list(map_task[task].values())[0] > best_task_map[task]:
                        best_task_map[task] = list(map_task[task].values())[0]
                        logger.info("Best {} map at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )

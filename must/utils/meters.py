#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

from copy import deepcopy
import datetime
from math import sqrt
import traceback
import numpy as np
import os
import json
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import must.evaluate.main_eval as grasp_eval
import must.utils.logging as logging
import must.utils.misc as misc

logger = logging.get_logger(__name__)

'''

To add a custom dataset, include the name of your dataset in the IDENT_FUNCT_DICT. This function is used to reconstruct the full name of a frame, 
formatted as {video}/{frame_name}, using the frame number and its corresponding name.

'''


IDENT_FUNCT_DICT = {
                    'grasp': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
                    'graspms': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
                    'Graspchunks': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),

                    'cholec80': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.jpg'.format(x,x,y),
                    'cholec80ms': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.jpg'.format(x,x,y),
                    'cholec80chunks': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.jpg'.format(x,x,y),

                    'misaw': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'misawms': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'Misawchunks': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),

                    'heichole': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    'heicholems': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    'Heicholechunks': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    }

class SurgeryMeterChunks(object):
    """
    Measure the PSI-AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        self.eval_train = cfg.TRAIN.EVAL_TRAIN

        self.tasks = deepcopy(cfg.TASKS.TASKS)
        self.log_tasks = deepcopy(cfg.TASKS.TASKS)
        self.metrics = deepcopy(cfg.TASKS.METRICS)
        
        self.online_inference = cfg.TEMPORAL_MODULE.ONLINE_INFERENCE

        self.all_classes = cfg.TASKS.NUM_CLASSES
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.task_loss = TaskMeter(cfg.LOG_PERIOD, len(self.log_tasks)) 
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = {k: [] for k in self.tasks}
        self.full_map = {}
        self.all_names = []
        self.overall_iters = overall_iters
        self.groundtruth = cfg.ENDOVIS_DATASET.TEST_COCO_ANNS

        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "lr": self.lr,
                "overall_loss": self.task_loss.get_tasks_median_sum(),
            }
            all_loss_medians = self.task_loss.get_win_median()
            for idx, task in enumerate(self.log_tasks):
                stats["loss_{}".format(task)] = all_loss_medians[idx]
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.task_loss.reset()
        self.full_map = {}
        self.all_preds = {task:[] for task in self.tasks}
        self.all_names = []

    def update_stats(self, preds, names, final_loss= None, losses=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            d_names (list): names of the keyframes with detection anns.
            names (list): names of all the keyframes.
            final_loss (float): final loss value.
            lr (float): learning rate.
        """ 
        if (self.eval_train or self.mode in ["val", "test"]) and not self.online_inference:
            for task in self.tasks:
                preds[task] = np.array(preds[task]).reshape(-1, np.array(preds[task]).shape[-1])
                self.all_preds[task].extend(preds[task])
            if self.parallel:
                names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in names]
            
            names = names.reshape(-1, 2)
            names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in names]
            
            self.all_names.extend(names)

            assert all(len(names)==len(preds[t]) for t in self.tasks)
        
        elif (self.eval_train or self.mode in ["val", "test"]) and self.online_inference:
            for task in self.tasks:
                preds_frame = np.array(preds[task])[:,-1,:]

                preds[task] = preds_frame
                self.all_preds[task].extend(preds[task])
            if self.parallel:
                names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in names]

            batch_size, seq_len, _= names.shape
            names = names[:,-1,:]
            names = [IDENT_FUNCT_DICT[self.dataset_name](*name) for name in names]
        
            self.all_names.extend(names)
            
            assert all(len(names)==len(preds[t]) for t in self.tasks)

        if losses is not None:
            self.task_loss.add_value(losses)
        if final_loss is not None:
            self.loss.add_value(final_loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, epoch, log=True):
        """
        Calculate and log the final PSI-AVA metrics.
        """
        out_name = {}
        for task,metric in zip(self.tasks, self.metrics):
            out_name[task] = self.save_json(task, self.all_preds, self.all_names, epoch)
            self.full_map[task] = grasp_eval.main_per_task(self.groundtruth, out_name[task], task, metric)
            if log:
                stats = {"mode": self.mode, "task": task, "metric": self.full_map[task]}
                logging.log_json_stats(stats)
        if log:
            stats = {"mode": self.mode, "mean metric": np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)])}
            logging.log_json_stats(stats)
        
        return self.full_map, np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)]), out_name
                    
    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            metrics_val, mean_map, out_files = self.finalize_metrics(cur_epoch +1)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            for idx, task in enumerate(self.tasks):
                stats["{}_map".format(task)] = self.full_map[task]

            logging.log_json_stats(stats)
            
            return metrics_val, mean_map, out_files

    def save_json(self, task, preds, names, epoch):
        """
        Save json for the specific task.
        Args:
            cur_epoch (int): the number of current epoch.
        """

        save_json_dict = {}
        save_json_list = {}
        assert len(preds[task])==len(names), f'Inconsistent lengths {len(preds[task])} {len(names)}'

        for idx, (pred, name) in enumerate(zip(preds[task], names)):
            task_key_name = f'{task}_score_dist'
            if name not in save_json_list:
                save_json_list[name] = [pred]
            else:
                save_json_list[name].append(pred)
        
        for name in save_json_list:
            scores = save_json_list[name]
            if len(scores) == 1:
                save_json_dict[name] = {task_key_name:scores[0].tolist()}
            else:
                scores = np.mean(scores, axis=0)
                save_json_dict[name] = {task_key_name:scores.tolist()}

        path_prediction = os.path.join(self.output_dir, f'epoch_{epoch}_preds_{task}.json')
        with open(path_prediction, "w") as outfile:  
            json.dump(save_json_dict, outfile) 
            
        return path_prediction
    
class SurgeryMeter(object):
    """
    Measure the PSI-AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        self.eval_train = cfg.TRAIN.EVAL_TRAIN

        self.tasks = deepcopy(cfg.TASKS.TASKS)
        self.log_tasks = deepcopy(cfg.TASKS.TASKS)
        self.metrics = deepcopy(cfg.TASKS.METRICS)

        self.all_classes = cfg.TASKS.NUM_CLASSES
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.task_loss = TaskMeter(cfg.LOG_PERIOD, len(self.log_tasks)) 
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = {k: [] for k in self.tasks}
        self.full_map = {}
        self.all_names = []
        self.overall_iters = overall_iters
        self.groundtruth = cfg.ENDOVIS_DATASET.TEST_COCO_ANNS

        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "lr": self.lr,
                "overall_loss": self.task_loss.get_tasks_median_sum(),
            }
            all_loss_medians = self.task_loss.get_win_median()
            for idx, task in enumerate(self.log_tasks):
                stats["loss_{}".format(task)] = all_loss_medians[idx]
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.task_loss.reset()
        self.full_map = {}
        self.all_preds = {task:[] for task in self.tasks}
        self.all_names = []

    def update_stats(self, preds, names, final_loss= None, losses=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            d_names (list): names of the keyframes with detection anns.
            names (list): names of all the keyframes.
            final_loss (float): final loss value.
            lr (float): learning rate.
        """ 
        if self.eval_train or self.mode in ["val", "test"]:
            
            for task in self.tasks:
                self.all_preds[task].extend(preds[task])
            
            if self.parallel:
                names = [IDENT_FUNCT_DICT[self.dataset_name.lower()](*name) for name in names]

            
            self.all_names.extend(names)

            assert all(len(names)==len(preds[t]) for t in self.tasks)

        if losses is not None:
            self.task_loss.add_value(losses)
        if final_loss is not None:
            self.loss.add_value(final_loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, epoch, log=True):
        """
        Calculate and log the final PSI-AVA metrics.
        """
        out_name = {}
        for task,metric in zip(self.tasks, self.metrics):
            out_name[task] = self.save_json(task, self.all_preds, self.all_names, epoch)
            self.full_map[task] = grasp_eval.main_per_task(self.groundtruth, out_name[task], task, metric)
            if log:
                stats = {"mode": self.mode, "task": task, "metric": self.full_map[task]}
                logging.log_json_stats(stats)
        if log:
            stats = {"mode": self.mode, "mean metric": np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)])}
            logging.log_json_stats(stats)
        
        return self.full_map, np.mean([v[m] for v,m in zip(list(self.full_map.values()), self.metrics)]), out_name
                    
    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            metrics_val, mean_map, out_files = self.finalize_metrics(cur_epoch +1)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            for idx, task in enumerate(self.tasks):
                stats["{}_map".format(task)] = self.full_map[task]

            logging.log_json_stats(stats)
            
            return metrics_val, mean_map, out_files

    def save_json(self, task, preds, names, epoch):
        """
        Save json for the specific task.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        save_json_dict = {}

        assert len(preds[task])==len(names), f'Inconsistent lengths {len(preds[task])} {len(names)}'
        
        for idx, (pred, name) in enumerate(zip(preds[task], names)):
            task_key_name = f'{task}_score_dist'

            save_json_dict[name] = {task_key_name:pred}

        path_prediction = os.path.join(self.output_dir, f'epoch_{epoch}_preds_{task}.json')
        with open(path_prediction, "w") as outfile:  
            json.dump(save_json_dict, outfile) 
            
        return path_prediction

class TaskMeter(object):
    """
    A task meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size, num_tasks):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.num_tasks = num_tasks
        self.task_meters = [ScalarMeter(window_size) for _ in range(num_tasks)]

    def reset(self):
        """
        Reset the individual meters.
        """
        [meter.reset() for meter in self.task_meters]

    def add_value(self, values):
        """
        Add a new scalar value to each of the task's deques.
        """
        [self.task_meters[idx].add_value(val.item()) for idx, val in enumerate(values)]

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
    
        return [np.median(meter.deque) for meter in self.task_meters]
    
    def get_tasks_median_avg(self):
        """
        """
        return np.mean(np.array(self.get_win_median()))
    
    def get_tasks_median_sum(self):
        """
        """
        return np.sum(np.array(self.get_win_median()))

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return [np.mean(meter.deque) for meter in self.task_meters]

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return [meter.total/meter.count for meter in self.task_meters]


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
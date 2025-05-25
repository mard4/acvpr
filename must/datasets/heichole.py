#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import logging
import numpy as np

from copy import deepcopy
from .surgical_dataset import SurgicalDataset, SurgicalDatasetChunks
from . import utils as utils
from .build import DATASET_REGISTRY

import torch
import json

from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
import glob

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Heichole(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "heichole"
        self.zero_fill = 5
        self.image_type = "png"
        self.cfg = cfg
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec 
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images,complete_name)

        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]} & {sec_idx} & {sec}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].split('_')[-1].replace('.'+self.image_type,''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+self.image_type,"")} {sec}'

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self._video_length, 
            online = self.cfg.DATA.ONLINE,
        )

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0
 
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks}

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        # Preprocess images and boxes
        imgs = self._images_and_boxes_preprocessing_cv2(
            imgs
        )
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name
        
        return imgs, all_labels, extra_data, frame_identifier



@DATASET_REGISTRY.register()
class Heicholems(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "heicholems"
        self.zero_fill = 5
        self.image_type = "png"
        self.cfg = cfg
        self.multi_sample_rate = cfg.DATA.MULTI_SAMPLING_RATE
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec 
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """

        # Get the path of the middle frame 
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type)

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images,complete_name)

        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]} & {sec_idx} & {sec}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].split('_')[-1].replace('.'+self.image_type,''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+self.image_type,"")} {sec}'

        sequence_pyramid = []

        sample_rate_set = self.multi_sample_rate

        for sample_rate in self.multi_sample_rate:
            seq_len = self._video_length * sample_rate
            seq = utils.get_sequence(
                center_idx,
                seq_len // 2,
                sample_rate,
                num_frames=len(self._image_paths[video_idx]),
                length = self._video_length,
                online = self.cfg.DATA.ONLINE,
            )

            sequence_pyramid.append(seq)

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])

        assert len(clip_label_list) > 0
        
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks} 

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]
                
        # Load images of current clip.
        images_pyramid = utils.process_sequences_parallel(
            sequence_pyramid,
            video_idx,
            self.cfg,
            self._image_paths,
            self._images_and_boxes_preprocessing_cv2
        )
        
        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('video_',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name

        extra_data = {}
        
        return images_pyramid, all_labels, extra_data, frame_identifier


@DATASET_REGISTRY.register()
class Heicholechunks(SurgicalDatasetChunks):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        super().__init__(cfg,split)
        self.dataset_name = "Heicholechunks"
        self.cfg = cfg
        self.zero_fill = 5
        self.image_type = "png"

        self.feature_paths = self.get_temporal_feature_paths_per_case(cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN)

        self._sample_rate = cfg.TEMPORAL_MODULE.SAMPLING_RATE
        self._video_length = cfg.TEMPORAL_MODULE.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        #breakpoint()
        return sec 
        
        
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """

        video_idx, sec_idx, chunk = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]

        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])

        seq_feats = chunk
        
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])

        assert len(clip_label_list) > 0

        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks} 
     
        image_paths = ['{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type) for sec in chunk]

        #image_paths = [self._image_paths[video_idx][frame] for frame in seq_feats]
        masked_num = chunk.count(-1)
        chunk_mask = [False] * (len(image_paths) - masked_num) + [True] * masked_num
        chunk_mask = np.array(chunk_mask, dtype=bool)

        fill_feats = []

        if masked_num != 0:
            image_paths = image_paths[:-masked_num]
            fill_feats = [[0.] * 3072 for i in range(masked_num)]

        for task in self._frame_tasks:
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        extra_data["sequence_mask"] = chunk_mask                        

        feature_paths = image_paths
        features = self._load_samples_features(feature_paths, self.cfg)

        temporal_features = torch.tensor(np.array(features))

        frame_identifier = []
        for frame in image_paths:
            sec = frame.split("/")[-1].split(".")[0]
            sec = int(sec.split("_")[-1])
            
            video_num = int(video_name.replace('video_',''))
            frame_identifier.append([video_num,sec]) 
        
        frame_identifier += [[0, 0]] * masked_num
        
        frame_identifier = np.array(frame_identifier)
        
        return temporal_features, all_labels, extra_data, frame_identifier
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import logging
import numpy as np

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY
import random

logger = logging.getLogger(__name__)


'''

To use your custom dataset with Detectron2, register it by adding the @DATASET_REGISTRY.register() decorator to your dataset class or function. Additionally, ensure that 
your custom dataset is imported in the __init__.py file within the datasets folder so it can be properly recognized during initialization.

'''

@DATASET_REGISTRY.register()
class YourDataset(SurgicalDataset):
    """
    YourDataset dataloader for MViT.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "yourdataset" # Name of your dataet
        self.zero_fill = 6 # The number of zeros your images are padded with
        self.image_type = "jpg" # The extension of your images (jpg, png, etc.)
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        pass # This function is only for the GraSP dataset
        
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

        # The complete name is how the name of your dataset is made {video_name}/{frame_number}.{jpg}. This depends on how the folders and frame names are.
        complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type) 

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images,complete_name)
        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]} & {sec_idx} & {sec}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
            length=self._video_length
        )

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        all_labels = {task:[] for task in self._region_tasks}

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]
                
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        imgs = self._images_and_boxes_preprocessing_cv2(
            imgs
        )
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('video','')) # For running in more than one gpu, you need to extract the number of your video
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name

        # Custom if want to add information
        extra_data = {}
        
        return imgs, all_labels, extra_data, frame_identifier


@DATASET_REGISTRY.register()
class YourDatasetpms(SurgicalDataset):
    """
    YourDataset multisequence dataloader for MTFE.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "yourdataset" # Name of your dataet
        self.zero_fill = 6 # The number of zeros your images are padded with
        self.image_type = "jpg" # The extension of your images (jpg, png, etc.)
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        pass # This function is only for the GraSP dataset

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

        sequence_pyramid = []

        sample_rate_set = self.multi_sample_rate

        for sample_rate in sample_rate_set:
            seq_len = self._video_length * sample_rate
            seq = utils.get_sequence(
                center_idx,
                seq_len // 2,
                sample_rate,
                num_frames=len(self._image_paths[video_idx]),
                length = self._seq_len
            )

            sequence_pyramid.append(seq)

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'

        # Get the frame idxs for current clip.
        images_pyramid = utils.process_sequences_parallel(
            sequence_pyramid,
            video_idx,
            self.cfg,
            self._image_paths,
            self._images_and_boxes_preprocessing_cv2
        )
        
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
            video_num = int(video_name.replace('video','')) # For running in more than one gpu, you need to extract the number of your video
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name
        
        # Custom if want to add information
        extra_data = {}

        return images_pyramid, all_labels, extra_data, frame_identifier

@DATASET_REGISTRY.register()
class YourDatasettransformer(Psi_ava_transformer):
    """
    YourDataset temporal consistency module dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "yourdatasettransformer" # Name of your dataet
        self.cfg = cfg
        self.zero_fill = 6 # The number of zeros your images are padded with
        self.image_type = "jpg" # The extension of your images (jpg, png, etc.)
        super().__init__(cfg,split)
    
    def keyframe_mapping(self, video_idx, sec_idx, sec):
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

        video_feat_paths = self.feature_paths[video_name]
        feat_idx = video_feat_paths.index(complete_name)
        
        seq_feats = utils.get_sequence(
            feat_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(video_feat_paths),
            length = self._seq_len
        )
        
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        all_labels = {task:[] for task in self._region_tasks} 
        
        feature_names = [video_feat_paths[frame] for frame in seq_feats]

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}      
        
        temporal_features = self._load_samples_features(feature_names, self.cfg)
        temporal_features = torch.Tensor(temporal_features)

        frame_identifier = complete_name  
        
        return temporal_features, all_labels, extra_data, complete_name
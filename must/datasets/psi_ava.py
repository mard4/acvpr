#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import logging
import numpy as np
import time

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY
import torch
import json

from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
import glob

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Psi_ava_transformer(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """
    def __init__(self, cfg, split, include_subvideo=True):
        self.cfg = cfg
        self.include_subvideo = include_subvideo

        self.fps = 1
        self.zero_fill = 9
        self.image_type = "jpg"

        self._split = split
        self._num_classes = {key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}

        self.dataset_name = "GraSP Transformer"

        self.fps_videos = {'CASE021','CASE041','CASE047','CASE050','CASE051','CASE053'}
        super().__init__(cfg, split)

        self.assignation = json.load(open("/media/lambda002/SSD6/srodriguezr2/endovis/tesis/TAPIS_official/TAPIS/association_30fps.json"))
        self.do_assignation = True if cfg.TRAIN.DATASET == "Psi_ava_transformer"  else False

        self.feature_paths = self.get_temporal_feature_paths_per_case(cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN)

        self._sample_rate = cfg.TEMPORAL_MODULE.SAMPLING_RATE
        self._video_length = cfg.TEMPORAL_MODULE.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate

    def keyframe_mapping(self, video_idx, sec_idx, sec):
        try:
            video_name = self._video_idx_to_name[video_idx]
            if video_name in self.fps_videos:
                return sec
            elif video_name=='CASE014':
                complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type)
                complete_path = os.path.join(self.cfg.ENDOVIS_DATASET.FRAME_DIR,complete_name)
                return self._image_paths[video_idx].index(complete_path)
            else:
                return round((sec*30)/45) 
        except:
            breakpoint()

    def _load_samples_features(self, samples, cfg, include_subvideo=False):
        if self._split == "train":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN
        elif self._split == "val":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_VAL

        output_features = []

        if self.do_assignation:
            inverted_assignation = {value: key for key, value in self.assignation.items()}
            key_list = list(inverted_assignation.keys())
        
        for img in samples:
            #if self.do_assignation:
                #img = inverted_assignation[img]
            
            if include_subvideo:
                case, subvideo, frame = img.split("/")[-3:]

                frame = f"{frame[:-4]}.pth"
                
                feature_path = os.path.join(feat_path, case, subvideo, frame)
            else:
                case, frame = img.split("/")[-2:]

                frame = f"{frame[:-4]}.pth"
                
                feature_path = os.path.join(feat_path, case, frame)

            feature_list = torch.load(feature_path)

            feat = np.concatenate(feature_list)

            output_features.append(feat.tolist())

        return output_features
    
    def _load_features(self, cfg):
        if self._split == "train":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_TRAIN
        elif self._split == "val":
            feat_path = cfg.TEMPORAL_MODULE.FEATURE_PATH_VAL

        videos_list = os.listdir(feat_path)
        feats_dict = {}

        for video in tqdm(videos_list, desc="Getting video features..."):
            feat_paths = glob.glob(os.path.join(feat_path,video, '*.pth'))
            for feature in feat_paths:
                feat_list = torch.load(feature)
                feat = np.concatenate(feat_list)

                case, img_name = feature.split("/")[-2:]
                img_key = f"{case}/{img_name[:-4]}.jpg"
                feats_dict[img_key] = feat
        
        self._feats_dict = feats_dict
    
    def _get_feature_path_names(self, image_paths):
        
        features_paths = []
        for original_path in image_paths:
            path_parts = original_path.split('/')
            case_folder = path_parts[-2]
            image_filename = path_parts[-1]
            new_path = os.path.join(case_folder, image_filename)
            features_paths.append(new_path)

        return features_paths

    def get_temporal_feature_paths_per_case(self, feature_paths):
        case_dict = {}

        for case in os.listdir(feature_paths):
            if case not in case_dict:
                case_dict[case] = []
            case_path = os.path.join(feature_paths, case)
        
            for frame in os.listdir(case_path):
                frame_path = os.path.join(case, frame).replace('pth', self.image_type)
                #if self.do_assignation:
                    #frame_path = self.assignation[frame_path]
                case_dict[case].append(frame_path)

        return case_dict


    def __getitem__(self, idx):
        
        # Get the path of the middle frame 
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
            
            video_num = int(video_name.replace('CASE',''))
            frame_identifier.append([video_num,sec]) 
        
        frame_identifier += [[0, 0]] * masked_num
        
        frame_identifier = np.array(frame_identifier)
        
        return temporal_features, all_labels, extra_data, frame_identifier


@DATASET_REGISTRY.register()
class Psi_ava(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """

    def __init__(self, cfg, split):
        self.dataset_name = "PSI-AVA"
        self.zero_fill = 5
        self.image_type = "jpg"
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

        #TODO: REMOVE when all done
        folder_to_images = "/".join(self._image_paths[video_idx][0].split('/')[:-2])
        path_complete_name = os.path.join(folder_to_images,complete_name)
        found_idx = self._image_paths[video_idx].index(path_complete_name)

        assert path_complete_name == self._image_paths[video_idx][center_idx], f'Different paths {path_complete_name} & {self._image_paths[video_idx][center_idx]}'
        assert found_idx == center_idx, f'Different indexes {found_idx} & {center_idx}'
        assert int(self._image_paths[video_idx][center_idx].split('/')[-1].replace('.'+self.image_type,''))==sec, f'Different {self._image_paths[video_idx][center_idx].split("/")[-1].replace("."+self.image_type,"")} {sec}'

        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        
        # Add labels depending on the task
        all_labels = {task:[] for task in self._region_tasks} 
        
 

        if self.cfg.REGIONS.ENABLE:
            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                all_labels_presence = {f'{task}_presence':np.zeros(self._num_classes[task]) for task in self._region_tasks}
                all_labels.update(all_labels_presence)
            for box_labels in clip_label_list:
                if box_labels['bbox'] != [0,0,0,0]:
                    boxes.append(box_labels['bbox'])
                    if self.cfg.FEATURES.ENABLE:
                        rpn_box_key = " ".join(map(str,box_labels['bbox']))
                        if rpn_box_key not in box_features[0].keys():
                            rpn_box_key = utils.get_best_features(box_labels["bbox"],box_features)
                        try:
                            features = np.array(box_features[0][rpn_box_key])
                            rpn_features.append(features)
                        except:

                            if self.cfg.FEATURES.MODEL=='detr':
                                # Deformable DETR extracted features size is 256
                                rpn_features.append(np.zeros(256))
                            elif self.cfg.FEATURES.MODEL=='m2f':
                                # Mask2Former extracted features size is 512
                                rpn_features.append(np.zeros(512))
                            elif self.cfg.FEATURES.MODEL=='faster':
                                # Faster-RCNN extracted features size is 1024
                                rpn_features.append(np.zeros(1024))
                                
                            # logger.info(f"=== No box features found for frame {path_complete_name} ===")

                    for task in self._region_tasks:
                        if isinstance(box_labels[task],list):
                            binary_task_label = np.zeros(self._num_classes[task],dtype='uint8')
                            box_task_labels = np.array(box_labels[task])-1
                            binary_task_label[box_task_labels] = 1
                            all_labels[task].append(binary_task_label)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_task_labels] = 1
                        elif isinstance(box_labels[task],int):
                            all_labels[task].append(box_labels[task]-1)
                            if self.cfg.TASKS.PRESENCE_RECOGNITION:
                                all_labels[f'{task}_presence'][box_labels[task]-1] = 1
                        else:
                            raise ValueError(f'Do not support annotation {box_labels[task]} of type {type(box_labels[task])} in frame {complete_name}')
            
        else:
            for task in self._region_tasks:
                binary_task_label = np.zeros(self._num_classes[task]+1, dtype='uint8')
                label_list = [label[task] for label in clip_label_list]
                assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inocnsistent label type {label_list} in frame {complete_name}'
                if isinstance(label_list[0], list):
                    label_list = set(list(itertools.chain(*label_list)))
                    binary_task_label[label_list] = 1
                elif isinstance(label_list[0], int):
                    label_list = set(label_list)
                    binary_task_label[label_list] = 1
                else:
                    raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {complete_name}')
                all_labels[task] = binary_task_label[1:]

        for task in self._frame_tasks:
            assert all(label[task]==clip_label_list[0][task] for label in clip_label_list), f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        extra_data = {}
        if self.cfg.REGIONS.ENABLE:
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self._split == 'train' else self.cfg.DATA.MAX_BBOXES
            if  len(boxes):
                ori_boxes = deepcopy(boxes)
                boxes = np.array(boxes)
                if self.cfg.FEATURES.ENABLE:
                    rpn_features = np.array(rpn_features)
            else:
                ori_boxes = []
                boxes = np.zeros((max_boxes, 4))
        else:
            boxes = np.zeros((1, 4))
                
        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )
        
        # Preprocess images and boxes
        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )
        
        # Padding and masking for a consistent dimensions in batch
        if self.cfg.REGIONS.ENABLE and len(ori_boxes):
            max_boxes = self.cfg.DATA.MAX_BBOXES * 2 if self._split == 'train' else self.cfg.DATA.MAX_BBOXES

            # TODO: REMOVE when all done
            assert len(boxes)==len(ori_boxes)==len(rpn_features), f'Inconsistent lengths {len(boxes)} {len(ori_boxes)} {len(rpn_features)}'
            assert len(boxes)<= max_boxes and len(ori_boxes)<=max_boxes and len(rpn_features)<=max_boxes, f'Incorrect lengths respect max box num{len(boxes)} {len(ori_boxes)} {len(rpn_features)}'

            bbox_mask = np.zeros(max_boxes,dtype=bool)
            bbox_mask[:len(boxes)] = True
            extra_data["boxes_mask"] = bbox_mask

            if len(boxes)<max_boxes:
                c_boxes = np.concatenate((boxes,np.zeros((max_boxes-len(boxes),4))),axis=0)
                boxes = c_boxes
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                if len(rpn_features)<max_boxes:
                    c_rpn_features = np.concatenate((rpn_features,np.zeros((max_boxes-len(rpn_features), 256 if self.cfg.FEATURES.MODEL=='detr' else (512 if self.cfg.FEATURES.MODEL=='m2f' else 1024)))),axis=0)
                    rpn_features = c_rpn_features
                extra_data["rpn_features"] = rpn_features
        elif self.cfg.REGIONS.ENABLE:
            bbox_mask = np.zeros(max_boxes,dtype=bool)
            extra_data["boxes_mask"] = bbox_mask
            extra_data["ori_boxes"] = ori_boxes
            extra_data["boxes"] = boxes

            if self.cfg.FEATURES.ENABLE:
                extra_data["rpn_features"] = np.zeros((max_boxes, 256 if self.cfg.FEATURES.MODEL=='detr' else (512 if self.cfg.FEATURES.MODEL=='m2f' else 1024)))
        
        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS>1:
            video_num = int(video_name.replace('CASE',''))
            frame_identifier = [video_num,sec]
        else:
            frame_identifier = complete_name
        
        return imgs, all_labels, extra_data, frame_identifier

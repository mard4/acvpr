#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import abstractmethod
import torch
import logging
import numpy as np
import os
import json

from . import surgical_dataset_helper as data_helper
from . import cv2_transform as cv2_transform
from . import utils as utils

logger = logging.getLogger(__name__)

class SurgicalDataset(torch.utils.data.Dataset):
    """
    We adapt the AVA Dataset management in Slowfast to manage Endoscopic Vision databases.
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        self._region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
        self._frame_tasks = {task for task in cfg.TASKS.TASKS if task not in cfg.ENDOVIS_DATASET.REGION_TASKS}

        # fix
        #assignation_path = self.cfg.ENDOVIS_DATASET.ASSIGNATION_FILE
        #self.assignation = json.load(open(assignation_path))

        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.ENDOVIS_DATASET.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.ENDOVIS_DATASET.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.ENDOVIS_DATASET.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.ENDOVIS_DATASET.TEST_FORCE_FLIP
        
        self._load_data(cfg)
    
    @abstractmethod
    def keyframe_mapping(self, video_idx, sec_idx, sec):
        pass

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = data_helper.load_image_lists(cfg, is_train=(self._split == "train"))

        # Loading annotations for boxes and labels.
        boxes_and_labels = data_helper.load_boxes_and_labels(
            cfg, mode=self._split
        )

        assert len(boxes_and_labels) == len(self._image_paths)
        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        if cfg.TEMPORAL_MODULE.CHUNKS==False:
            (
                self._keyframe_indices,
                self._keyframe_boxes_and_labels,
            ) = data_helper.get_keyframe_data(boxes_and_labels, self.keyframe_mapping)
        else:
            (
                self._keyframe_indices,
                self._keyframe_boxes_and_labels,
            ) = data_helper.get_keyframe_data_chunks(boxes_and_labels, self.keyframe_mapping, self.cfg, self._split)

                # Calculate the number of used boxes.
        self._num_boxes_used = data_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        logger.info(f"=== {self.dataset_name} dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.

        Returns:
            imgs (tensor): list of preprocessed images.
        """

        height, width, _ = imgs[0].shape

        # The image now is in HWC, BGR format.
        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:  # "train"
            if self.cfg.DATA.FIXED_RESIZE:
                imgs = [cv2_transform.scale_resize(250, img) for img in imgs]

            else:
                imgs = cv2_transform.random_short_side_scale_jitter_list(
                    imgs,
                    min_size=self._jitter_min_scale,
                    max_size=self._jitter_max_scale,
                )

            imgs = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC"
            )

            if self.random_horizontal_flip:
                # random flip
                imgs = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC"
                )
        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            # Short side to test_scale. Non-local and STRG uses 256.
            if self.cfg.DATA.FIXED_RESIZE:
                imgs = [cv2_transform.scale_resize(250, img) for img in imgs]
            else:
                imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]

            imgs= cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1
            )

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                imgs = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC"
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]


        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)

        return imgs

class SurgicalDatasetChunks(SurgicalDataset):
    """
    PSI-AVA dataloader.
    """
    def __init__(self, cfg, split, include_subvideo=True):
        self.cfg = cfg
        self.include_subvideo = include_subvideo
        
        self.dataset_name = cfg.TRAIN.DATASET

        self.fps = 1
        self.zero_fill = 9
        self.image_type = "jpg"

        self._split = split
        self._num_classes = {key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        
        print(cfg)   
        # pick up the association file (may be empty string for datasets that don't use it)
        assignation_path = cfg.ENDOVIS_DATASET.ASSIGNATION_FILE
        if assignation_path:
            self.assignation = json.load(open(assignation_path))
        else:
            self.assignation = {}

            
        self.do_assignation = cfg.TRAIN.DATASET.endswith("chunks")

        self.fps_videos = {'CASE021','CASE041','CASE047','CASE050','CASE051','CASE053'}
        super().__init__(cfg, split)

        #self.do_assignation = True if cfg.TRAIN.DATASET == f"{self.dataset_name}chunks"  else False

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

                case_dict[case].append(frame_path)

        return case_dict
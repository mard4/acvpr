#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import math

from .cross_vit import *
import os
import json
from .backbones import ConvTransformerBackbone

from .utils import PositionalEncoding


IDENT_FUNCT_DICT = {
                    'grasp': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),
                    'graspms': lambda x,y: 'CASE{:03d}/{:09d}.jpg'.format(x,y),

                    'cholec80': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.jpg'.format(x,x,y),
                    'cholec80ms': lambda x,y: 'video{:02d}/video{:02d}_{:06d}.jpg'.format(x,x,y),

                    'misaw': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),
                    'misawms': lambda x,y: 'CASE{:03d}/{:05d}.jpg'.format(x,y),

                    'heichole': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    'heicholems': lambda x,y: 'video_{:02d}/{:05d}.png'.format(x,y),
                    }


class TransformerBasicHead(nn.Module):
    """
    Frame Classification Head of TAPIS.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
        recognition=False
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.class_projection = nn.Linear(dim_in, num_classes, bias=True)
        self.cls_embed = cls_embed
        self.recognition = recognition
        self.act_func = act_func

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x, features=None, boxes_mask=None):
        if self.cls_embed and not self.recognition:
            x = x[:, 0]
        elif self.cls_embed:
            x = x[:,1:].mean(1)
        else:
            x = x.mean(1)

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.class_projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)
        return x


class CrossAttentionModule(nn.Module):
    """
    This class supports:
    - Multi-sequence cross-attention
    - Full sequence self-attention
    - Configurable activation functions
    - MLP-based logit joining
    - Dropout regularization

    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cls_embed=False,
    ):
        """
        Initialize the multi-sequence transformer head.
        Args:
            cfg: Configuration object containing model hyperparameters
            dim_in: Input feature dimension
            num_classes: Number of output classes
            dropout_rate: Probability of dropout (default: 0.0)
            act_func: Activation function type (default: 'softmax')
            cls_embed: Flag for classification embedding (default: False)
        """
        super(CrossAttentionModule, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        self.dataset_name = cfg.TEST.DATASET
        self.parallel = cfg.NUM_GPUS > 1
        self.act_func = act_func
        self.multiscale_encoder = nn.ModuleList([])
        self.full_sequence_self_attention = nn.ModuleList([])

        self.mvit_feats_enable = cfg.MVIT_FEATS.ENABLE
        self.mvit_feats_path = cfg.MVIT_FEATS.PATH
        self.self_attn_layers = cfg.MULTISCALEATTN.SELF_ATTN_LAYERS

        self.num_sequences = len(cfg.DATA.MULTI_SAMPLING_RATE)

        # Initialize MultiScale Attention for each level of the pyramid
        for _ in range(self.num_sequences):
            self.multiscale_encoder.append(MultiScaleEncoder(
                                    depth=1,
                                    sm_dim=cfg.MULTISCALEATTN.CROSS_ATTN_EMBED_DIM,
                                    lg_dim=cfg.MULTISCALEATTN.CROSS_ATTN_EMBED_DIM,
                                    cross_attn_depth=cfg.MULTISCALEATTN.CROSS_ATTN_DEPTH,
                                    cross_attn_heads=cfg.MULTISCALEATTN.CROSS_ATTN_HEADS,
                                    cross_attn_dim_head=cfg.MULTISCALEATTN.CROSS_ATTN_DIM_HEAD,
                                    dropout=0.1
                                    ))
        # Initialize Self-Attention 
        for _ in range(self.num_sequences):
            if self.self_attn_layers > 1:
                # Create a list of self-attention layers for each sequence
                self_attn_layers = nn.ModuleList([
                    torch.nn.MultiheadAttention(
                        embed_dim=cfg.MULTISCALEATTN.SELF_ATTN_EMBED_DIM,
                        num_heads=cfg.MULTISCALEATTN.SELF_ATTN_NUM_HEADS,
                        batch_first=True,
                        dropout=0.1
                    )
                    for _ in range(self.self_attn_layers)
                ])
                self.full_sequence_self_attention.append(self_attn_layers)
            else:
                # Add a single self-attention layer for each sequence
                self.full_sequence_self_attention.append(
                    torch.nn.MultiheadAttention(
                        embed_dim=cfg.MULTISCALEATTN.SELF_ATTN_EMBED_DIM,
                        num_heads=cfg.MULTISCALEATTN.SELF_ATTN_NUM_HEADS,
                        batch_first=True,
                        dropout=0.1
                    )
                )
            
        input_size = cfg.MULTISCALEATTN.SELF_ATTN_EMBED_DIM * self.num_sequences
        # MLP that processes the embeddings from each level
        self.mlp_logits_embedding = nn.Sequential(
                                        nn.Dropout(p=0.1),
                                        nn.Linear(input_size, input_size, bias=True),
                                    )
        # Classification head
        self.mlp_classifier = nn.Sequential(
                                        nn.Tanh(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(input_size, num_classes, bias=True)
                                    )

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def save_cls_tokens(self, x, image_names):
        json_data = {}
        if self.parallel:
            image_names = [IDENT_FUNCT_DICT[self.dataset_name.lower()](*name) for name in image_names]
        
        for idx, frame_name in enumerate(image_names):
            if frame_name in json_data:
                name = json_data[frame_name]
            else:
                name = frame_name
            
            mvit_feats_dictionary = []
            video_name = name.split('/')[0]
            if not os.path.exists(os.path.join(self.mvit_feats_path, video_name)):
                os.makedirs(os.path.join(self.mvit_feats_path, video_name))
                
            sequence_embeddings = x[idx].data.cpu().numpy()
            mvit_feats_dictionary.extend([sequence_embeddings])
            feat_name = name.split('.')[0] + '.pth'
            path = os.path.join(self.mvit_feats_path, feat_name)
            torch.save(mvit_feats_dictionary, path)

    def upload_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    

    def forward(self, x, image_names=None):
        b_size, seq_dim, embed_dim = x[0].shape
        all_sequences = x

        sequences = []

        # Prepare sequences
        for idx in range(self.num_sequences):
            main_seq = all_sequences[idx]
            other_seqs = all_sequences[:idx] + all_sequences[idx+1:]
            sequences.append((main_seq, tuple(other_seqs)))

        embeddings = []
        tokens = torch.zeros((b_size, len(x), embed_dim)).cuda()
        
        # Perform cross-attention for each level of the pyramid
        for idx, (seq_tokens, context) in enumerate(sequences):

            encoded_seq = self.multiscale_encoder[idx](seq_tokens, context)
            context = torch.cat(context, dim=1)
            encoded_seq = torch.cat((encoded_seq, seq_tokens), dim=1)
            if self.self_attn_layers > 1:
                for idx_self_attn in range(self.self_attn_layers):
                    encoded_seq = self.full_sequence_self_attention[idx][idx_self_attn](encoded_seq, 
                                                                        encoded_seq,
                                                                        encoded_seq)[0]  
            else:
                encoded_seq = self.full_sequence_self_attention[idx](encoded_seq, 
                                                                    encoded_seq,
                                                                    encoded_seq)[0]
        
            cls_token = encoded_seq[:, 0]

            tokens[:, idx, :] = cls_token

            embeddings.append(cls_token)
        # Join the logits from each level 
        embeddings = torch.stack(embeddings).cuda()
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        # Fuse the logits from each level
        fused_embeddings = self.mlp_logits_embedding(embeddings)
        logits = self.mlp_classifier(fused_embeddings)
        
        if self.act_func == "sigmoid" or not self.training:
            x = self.act(logits)

        if image_names is not None:
            self.save_cls_tokens(fused_embeddings, image_names)

        return logits


class ClassificationBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ClassificationBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        self.act_func = act_func
        self.cfg = cfg

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            dim = 2

            if cfg.TEMPORAL_MODULE.CHUNKS == False:
                dim = 1

            self.act = nn.Softmax(dim=dim)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)

        return x

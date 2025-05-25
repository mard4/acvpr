# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

from copy import deepcopy
import math
from functools import partial
import torch
import torch.nn as nn
import os
import json
import numpy as np
import copy

import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import must.utils.weight_init_helper as init_helper
from .attention import MultiScaleBlock
from .attentionv2 import MultiScaleBlock as MultiScaleBlockv2
from .batchnorm_helper import get_norm
from .utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import, 
    PositionalEncoding
)

from . import head_helper, resnet_helper, stem_helper, stem_helperv2
from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None

from .backbones import ConvTransformerBackbone

_POOL1 = {
    "mvit": [[2, 1, 1]], 
    'MMViT': [[2, 1, 1]],
}


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        
        # Prepare PSI-AVA tasks
        self.tasks = deepcopy(cfg.TASKS.TASKS)
        self.num_classes = deepcopy(cfg.TASKS.NUM_CLASSES)
        self.act_fun = deepcopy(cfg.TASKS.HEAD_ACT)
        self.recogn = cfg.TASKS.PRESENCE_RECOGNITION

            
        # Prepare output.
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        # Following MocoV3, initializing with random patches stabilize optimization
        if cfg.MVIT.FREEZE_PATCH:
            self.patch_embed.requires_grad = False
            
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, embed_dim)
                )
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()

        if cfg.MODEL.ACT_CHECKPOINT:
            validate_checkpoint_wrapper_import(checkpoint_wrapper)
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
            )
            if cfg.MODEL.ACT_CHECKPOINT:
                attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

        self.embed_dim = dim_out
        self.norm = norm_layer(self.embed_dim)
        pool_size = _POOL1[cfg.MODEL.ARCH]
        pool_size[0][0] = self.patch_stride[0]

        self.mvit_feats_enable = cfg.MVIT_FEATS.ENABLE
        self.mvit_feats_path = cfg.MVIT_FEATS.PATH

        for idx, task in enumerate(self.tasks):
            extra_head = head_helper.TransformerBasicHead(
                        self.embed_dim,
                        self.num_classes[idx],
                        dropout_rate=cfg.MODEL.DROPOUT_RATE,
                        act_func=self.act_fun[idx],
                        cls_embed=self.cls_embed_on,
                        recognition=False
                    )
            
            self.add_module("extra_heads_{}".format(task), extra_head)
   
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def upload_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data


    def forward(self, x):
        out = {}
        x = x[0].cuda()
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)

        # MuST head classification
        for task in self.tasks:
            extra_head = getattr(self, "extra_heads_{}".format(task))
            out[task] = extra_head(x)
                
        return out

import time



@MODEL_REGISTRY.register()
class MMViT(MViT):
    """
    Multi-term frame encoder
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Get parameters.
        for idx, task in enumerate(self.tasks):
        
            extra_head = head_helper.CrossAttentionModule(
                        cfg,
                        self.embed_dim,
                        self.num_classes[idx],
                        dropout_rate=cfg.MODEL.DROPOUT_RATE,
                        act_func=self.act_fun[idx],
                        cls_embed=self.cls_embed_on,
                    )
            

            self.add_module("extra_heads_{}".format(task), extra_head)
   
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            if self.cls_embed_on:
                trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)


    def forward(self, x_seq, image_names=None):
        # breakpoint()
        out = {}
        outs = []
        for x in x_seq:
            x = x.cuda()
            x = self.patch_embed(x)

            T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
            H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
            W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
            B, N, C = x.shape

            if self.cls_embed_on:
                cls_tokens = self.cls_token.expand(
                    B, -1, -1
                )  # stole cls_tokens impl from Phil Wang, thanks
                x = torch.cat((cls_tokens, x), dim=1)

            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

            if self.drop_rate:
                x = self.pos_drop(x)

            if self.norm_stem:
                x = self.norm_stem(x)

            thw = [T, H, W]
            for blk in self.blocks:
                x, thw = blk(x, thw)

            x = self.norm(x)
            outs.append(x)

        # MuST head classification
        for task in self.tasks:
            extra_head = getattr(self, "extra_heads_{}".format(task))
            out[task] = extra_head(outs, image_names)

        return out


@MODEL_REGISTRY.register()
class TCM(nn.Module):
    def __init__(self, cfg, classifier=True, max_len=1000):
        super(TCM, self).__init__()
        self.tasks = cfg.TASKS.TASKS
        self.num_classes = cfg.TASKS.NUM_CLASSES
        self.act_fun = cfg.TASKS.HEAD_ACT
        self.seq_len = cfg.TEMPORAL_MODULE.NUM_FRAMES

        self.embedding = nn.Linear(cfg.TEMPORAL_MODULE.TCM_INPUT_DIM, cfg.TEMPORAL_MODULE.TCM_D_MODEL)

        input_dim = cfg.TEMPORAL_MODULE.TCM_INPUT_DIM
        d_model = cfg.TEMPORAL_MODULE.TCM_D_MODEL

        self.positional_encoding = PositionalEncoding(cfg.TEMPORAL_MODULE.TCM_D_MODEL, max_len)

        encoder_layer = nn.TransformerEncoderLayer(cfg.TEMPORAL_MODULE.TCM_D_MODEL, cfg.TEMPORAL_MODULE.TCM_NUM_HEADS)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.TEMPORAL_MODULE.TCM_NUM_LAYERS)
    

        self.classifier = classifier

        if classifier:

            for idx, task in enumerate(self.tasks):
                extra_head = head_helper.ClassificationBasicHead(
                        cfg,
                        cfg.TEMPORAL_MODULE.TCM_D_MODEL,
                        self.num_classes[idx],
                        dropout_rate=cfg.MODEL.DROPOUT_RATE,
                        act_func=self.act_fun[idx],
                        )
            
                self.add_module("extra_heads_{}".format(task), extra_head)
        
    def forward(self, x, features=None, boxes_mask=None, sequence_mask=None):
        out = {}
        
        x = x.cuda().float()

        x = self.embedding(x)
        x = self.positional_encoding(x)

        x = x.permute(1, 0, 2)

        x = self.encoder(x)
        x = x.permute(1, 0, 2)

        for task in self.tasks:
            extra_head = getattr(self, "extra_heads_{}".format(task))
            out[task] = extra_head(x)

        return out

# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmdet.models.decode_heads.isa_head import ISALayer
from mmdet.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..builder import build_loss
from .aspp_head import ASPPModule
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
from mmseg.models.losses import accuracy


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHeadDevil2(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHeadDevil2, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        loss_cls_clip=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.loss_clip = build_loss(loss_cls_clip)
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)
        self.debug_output = {}

    def forward_train(self,
                      inputs,
                      feats_gt,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #breakpoint()
        seg_logits,features = self.forward(inputs)
        import torch.nn.functional as F
        breakpoint()
        features = F.interpolate(features, [features.shape[2]*4, features.shape[3]*4], mode='bilinear', align_corners=True)
        B,C,H,W=features.shape
        N=20
        features = features.permute(0,2,3,1).reshape(1,H*W*B,C)
        expanded_predicted_embeddings = features.expand(N,H*W*B,C)
        expanded_predicted_embeddings=expanded_predicted_embeddings.reshape(N,H*W*B*C)
        #expanded_groundtruth_embeddings= expanded_groundtruth_embeddings.permute(2,0,1)
        #predicted_embeddings = predicted_embeddings.permute(0, 2, 3, 1).contiguous()
        #predicted_embeddings = predicted_embeddings.view(-1, embedding_size)
        #predicted_embeddings_expand=predicted_embeddings.expand(B*C*H*W,C,N)
        #predicted_embeddings_expand=predicted_embeddings_expand.permute(0,2, 1)
        expanded_groundtruth_embeddings=feats_gt.reshape(1,N,C).expand(H*W*B,N,C)
        expanded_groundtruth_embeddings=expanded_groundtruth_embeddings.reshape(N,H*W*B*C)
        similarity_loss = cosine_similarity_loss(expanded_predicted_embeddings, expanded_groundtruth_embeddings, torch.ones(expanded_predicted_embeddings.size(1)))
        log_softmax_predictions = F.log_softmax(similarity_loss, dim=1)
        nll_loss_value = nll_loss(log_softmax_predictions, gt_semantic_seg)
        #features = F.interpolate(features, [features.shape[2]*4, features.shape[3]*4], mode='bilinear', align_corners=True)
        #breakpoint()
        #features = torch.permute(features, (1, 0, 2, 3))
        #breakpoint()
        
        #features=features.reshape(768, features.shape[1]*features.shape[2]*features.shape[3])
        #breakpoint()
        #feats_gt=F.normalize(feats_gt, p=2, dim=1)
        #features=F.normalize(features, p=2, dim=0)
        #sim =feats_gt@features

        #todo: make sure this is correct
        #soft_score=sim#F.softmax(sim, dim=0)
        #breakpoint()
        
        #breakpoint()
        #loss= torch.nn.CrossEntropyLoss()
        #input=soft_score.permute(1,0).to('cuda')
        #target=torch.flatten(gt_semantic_seg).to('cuda')
        #target[target==255] = 19
        #s=loss(input,target)
        
        
        losses = self.losses(seg_logits, gt_semantic_seg, nll_loss_value)
        #breakpoint()
        return losses

    def losses(self, seg_logit, seg_label, seg_weight=None,loss_cls_=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        #breakpoint()
        loss_cls_ = nll_loss_value
        loss['loss_clip'] = loss_cls_ 
        #breakpoint()                                   
        return loss
    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        os_size = x[0].size()[2:]
        _c = {}

        #(Pdb) inputs[0].shape
        #torch.Size([2, 64, 128, 128])
        #(Pdb) inputs[1].shape
        #torch.Size([2, 128, 64, 64])
        #(Pdb) inputs[2].shape
        #torch.Size([2, 320, 32, 32])
        #(Pdb) inputs[3].shape
        #torch.Size([2, 512, 16, 16])
        #(Pdb) inputs[4].shape
        #breakpoint()
        for i in self.in_index:
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if _c[i].size()[2:] != os_size:
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)
        
        x_ = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        #breakpoint()
        x = self.cls_seg(x_)
        self.debug_output.update({'semantic': x.detach()})

        return x,x_

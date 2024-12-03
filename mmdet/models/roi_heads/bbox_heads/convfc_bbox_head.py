# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared2FCBBoxHeadCONF(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHeadCONF, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
    def _get_target_single(self,
                           pos_bboxes,
                           neg_bboxes,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           scores_instance,
                           pos_inds, #
                           pos_assigned_gt_inds,
                           gt_box_domain_indicator,
                           pseudo_wght_val,
                           cfg):
        '''

        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_inds_list,
            pos_assigned_gt_inds_list,
            gt_box_domain_indicator_list,
            pseudo_wght_val_list,
            cfg=rcnn_train_cfg)
        '''
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        #breakpoint()

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        #breakpoint()
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            #breakpoint()
            label_weights[:num_pos] = scores_instance[pos_assigned_gt_inds].to('cuda')#Elham
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = scores_instance[pos_assigned_gt_inds].repeat(4,1).permute(1,0).to('cuda')#Elham1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        # -------------------------------------------------------------------------------------------
        # --- added by Suman on  Oct 19th 2022  ---
        # -------------------------------------------------------------------------------------------
        if gt_box_domain_indicator is not None:
            # print('gt_box_domain_indicator')
            # print(gt_box_domain_indicator)
            # print(f'gt_box_domain_indicator.shape={gt_box_domain_indicator.shape}')
            # print(f'pos_inds.shape[0]={pos_inds.shape[0]}')
            # print(f'num_samples:{num_samples} = num_pos:{num_pos} + num_neg:{num_neg}')
            # print('pos_assigned_gt_inds')
            # print(pos_assigned_gt_inds)
            # print('---')
            for i in range(pos_inds.shape[0]):
                if gt_box_domain_indicator[pos_assigned_gt_inds[i]] == 0:  # i-th positive GT box belongs to source domain,
                    continue
                assert gt_box_domain_indicator[pos_assigned_gt_inds[i]] == 1, 'domain value must be 1 for target boxes!'
                bbox_weights[i, :] = pseudo_wght_val
                label_weights[i] = pseudo_wght_val
            #     print(f'i:{i} :pos_assigned_gt_inds[i]: {pos_assigned_gt_inds[i]} : pseudo_wght_val:{pseudo_wght_val}')
            # print('-----------------------')
        # -------------------------------------------------------------------------------------------
        #breakpoint()
        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    gt_box_domain_indicator=None,
                    pseudo_wght_val=None,
                    scores_instance=None
                    ):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        gt_box_domain_indicator_list = gt_box_domain_indicator
        pseudo_wght_val_list = pseudo_wght_val
        #breakpoint()

        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]

        if gt_box_domain_indicator_list is not None:
            pos_inds_list = [res.pos_inds for res in sampling_results]                          # added by Suman on 19 Oct 2022
            pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results] # added by Suman on 19 Oct 2022
        else:
            num_imgs = len(sampling_results)
            pos_inds_list =  [None for _ in range(num_imgs)]
            #pos_assigned_gt_inds_list = [None for _ in range(num_imgs)] Elham
            gt_box_domain_indicator_list = [None for _ in range(num_imgs)]
            pseudo_wght_val_list = [None for _ in range(num_imgs)]

        
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            scores_instance,
            pos_inds_list,
            pos_assigned_gt_inds_list,
            gt_box_domain_indicator_list,
            pseudo_wght_val_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        #breakpoint()
        return labels, label_weights, bbox_targets, bbox_weights

@HEADS.register_module()
class Shared2FCBBoxHeadPosOnly(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHeadPosOnly, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

    def _get_target_single(self,
                            pos_bboxes,
                            neg_bboxes,
                            pos_gt_bboxes,
                            pos_gt_labels,
                            pos_inds, #
                            pos_assigned_gt_inds,
                            gt_box_domain_indicator,
                            pseudo_wght_val,
                            cfg):
            '''

            labels, label_weights, bbox_targets, bbox_weights = multi_apply(
                self._get_target_single,
                pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                pos_inds_list,
                pos_assigned_gt_inds_list,
                gt_box_domain_indicator_list,
                pseudo_wght_val_list,
                cfg=rcnn_train_cfg)
            '''
            """Calculate the ground truth for proposals in the single image
            according to the sampling results.

            Args:
                pos_bboxes (Tensor): Contains all the positive boxes,
                    has shape (num_pos, 4), the last dimension 4
                    represents [tl_x, tl_y, br_x, br_y].
                neg_bboxes (Tensor): Contains all the negative boxes,
                    has shape (num_neg, 4), the last dimension 4
                    represents [tl_x, tl_y, br_x, br_y].
                pos_gt_bboxes (Tensor): Contains gt_boxes for
                    all positive samples, has shape (num_pos, 4),
                    the last dimension 4
                    represents [tl_x, tl_y, br_x, br_y].
                pos_gt_labels (Tensor): Contains gt_labels for
                    all positive samples, has shape (num_pos, ).
                cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

            Returns:
                Tuple[Tensor]: Ground truth for proposals
                in a single image. Containing the following Tensors:

                    - labels(Tensor): Gt_labels for all proposals, has
                    shape (num_proposals,).
                    - label_weights(Tensor): Labels_weights for all
                    proposals, has shape (num_proposals,).
                    - bbox_targets(Tensor):Regression target for all
                    proposals, has shape (num_proposals, 4), the
                    last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                    - bbox_weights(Tensor):Regression weights for all
                    proposals, has shape (num_proposals, 4).
            """
            num_pos = pos_bboxes.size(0)
            num_neg = neg_bboxes.size(0)
            num_samples = num_pos + num_neg

            # original implementation uses new_zeros since BG are set to be 0
            # now use empty & fill because BG cat_id = num_classes,
            # FG cat_id = [0, num_classes-1]
            labels = pos_bboxes.new_full((num_samples, ),
                                        self.num_classes,
                                        dtype=torch.long)
            label_weights = pos_bboxes.new_zeros(num_samples)
            bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
            bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
            if num_pos > 0:
                labels[:num_pos] = pos_gt_labels
                pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
                label_weights[:num_pos] = pos_weight
                if not self.reg_decoded_bbox:
                    pos_bbox_targets = self.bbox_coder.encode(
                        pos_bboxes, pos_gt_bboxes)
                else:
                    # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                    # is applied directly on the decoded bounding boxes, both
                    # the predicted boxes and regression targets should be with
                    # absolute coordinate format.
                    pos_bbox_targets = pos_gt_bboxes
                bbox_targets[:num_pos, :] = pos_bbox_targets
                bbox_weights[:num_pos, :] = 1
            #if num_neg > 0:
            #    label_weights[-num_neg:] = 1.0
            # -------------------------------------------------------------------------------------------
            # --- added by Suman on  Oct 19th 2022  ---
            # -------------------------------------------------------------------------------------------
            if gt_box_domain_indicator is not None:
                # print('gt_box_domain_indicator')
                # print(gt_box_domain_indicator)
                # print(f'gt_box_domain_indicator.shape={gt_box_domain_indicator.shape}')
                # print(f'pos_inds.shape[0]={pos_inds.shape[0]}')
                # print(f'num_samples:{num_samples} = num_pos:{num_pos} + num_neg:{num_neg}')
                # print('pos_assigned_gt_inds')
                # print(pos_assigned_gt_inds)
                # print('---')
                for i in range(pos_inds.shape[0]):
                    if gt_box_domain_indicator[pos_assigned_gt_inds[i]] == 0:  # i-th positive GT box belongs to source domain,
                        continue
                    assert gt_box_domain_indicator[pos_assigned_gt_inds[i]] == 1, 'domain value must be 1 for target boxes!'
                    bbox_weights[i, :] = pseudo_wght_val
                    label_weights[i] = pseudo_wght_val
                #     print(f'i:{i} :pos_assigned_gt_inds[i]: {pos_assigned_gt_inds[i]} : pseudo_wght_val:{pseudo_wght_val}')
                # print('-----------------------')
            # -------------------------------------------------------------------------------------------
            #breakpoint()
            return labels, label_weights, bbox_targets, bbox_weights


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

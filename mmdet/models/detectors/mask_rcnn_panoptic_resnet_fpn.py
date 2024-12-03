# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from mmdet.ops import resize
from mmseg.core import add_prefix
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.utils import get_root_logger
import torch.nn as nn

from .untils import tokenize
import os




@DETECTORS.register_module()
class DenseCLIP_MaskRCNNFPN(TwoStageDetector):
    """
    DenseCLIP for Mask-RCNN
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 decode_head,
                 context_length,
                 class_names,
                 seg_loss=False,
                 clip_head=True,
                 text_adapter=None,
                 tau=0.07,
                 token_embed_dim=512, 
                 text_dim=1024,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_neck_feat_for_decode_head=False,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)

        if neck is not None:
            self.neck = build_neck(neck)
        
        self._init_decode_head(decode_head)
        #self._init_auxiliary_head(auxiliary_head)
        self.use_neck_feat_for_decode_head = use_neck_feat_for_decode_head
        #breakpoint()

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.context_length = context_length
        self.tau = tau
        # add a [background] class
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.use_seg_loss = seg_loss
        self.class_names = class_names
        self.clip_head = clip_head
        #breakpoint()

        # learnable textual contexts
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        if self.train_cfg is not None:
            self.folder=self.train_cfg['folder']#'diff_roi_hyperbumble_t0_testHypers_RoiScale_where_scale56_bothWeighted0'
        else:
            self.folder=self.test_cfg['folder']
        self.count=0
        self.test_count=0
        self.patt=1000
        os.makedirs('PATH_TO_FILL/'+self.folder+'_train', exist_ok=True)
        os.makedirs('PATH_TO_FILL/'+self.folder+'_test', exist_ok=True)


    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        #from mmseg.models import builder
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        # init daformer decode head

    #def _init_decode_head(self, decode_head):
    #    """Initialize ``decode_head``"""
    #    self.decode_head = build_head(decode_head)
    #    self.align_corners = self.decode_head.align_corners
    #    self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""
        x_b = self.backbone(img)
        # (Pdb) x[0].shape
        # torch.Size([2, 256, 128, 128])
        # (Pdb) x[1].shape
        # torch.Size([2, 512, 64, 64])
        # (Pdb) x[2].shape
        # torch.Size([2, 1024, 32, 32])
        # (Pdb) x[3].shape
        # torch.Size([2, 2048, 16, 16])
        #len(x[4]) 2
        #breakpoint()
        text_features = self.compute_text_features(x_b, dummy=dummy)
        score_maps = self.compute_score_maps(x_b, text_features)
        #breakpoint()
        x_b = list(x_b[:-1])
        # (Pdb) x[0].shape
        # torch.Size([2, 256, 128, 128])
        # (Pdb) x[1].shape
        # torch.Size([2, 512, 64, 64])
        # (Pdb) x[2].shape
        # torch.Size([2, 1024, 32, 32])
        # (Pdb) x[3].shape
        # torch.Size([2, 2048, 16, 16])

        x_b[3] = torch.cat([x_b[3], score_maps[3]], dim=1) #torch.Size([2, 2067, 16, 16])
        #breakpoint()
        #breakpoint()
        #[2,3,512,512]
        #torch.Size([2, 256, 128, 128])
        #(Pdb) x[1].shape
        #torch.Size([2, 256, 64, 64])
        #(Pdb) x[2].shape
        #torch.Size([2, 256, 32, 32])
        #(Pdb) x[3].shape
        #torch.Size([2, 256, 16, 16])
        #(Pdb) x[4].shape
        #torch.Size([2, 256, 8, 8])
        x=x_b
        if self.with_neck:
            x = self.neck(x)
        #breakpoint()
        if use_seg_loss:
            #breakpoint()
            return x_b, x, score_maps[0]
        else:
            #breakpoint()
            return x

    def compute_score_maps(self, x, text_features):
        # B, K, C
        _, visual_embeddings = x[4]
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map3 = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_features) / self.tau
        score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone, 
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape

        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        else:
            text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, dummy=True)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, seg_weight=None):
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg, seg_weight)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    def _decode_head_forward_test(self, x, img_metas):
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits
    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None,
    #                   gt_masks=None,
    #                   proposals=None,
    #                   **kwargs):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      box_domain_indicator=None,
                      pseudo_wght_val=None,
                      activate_visual_debug=False,
                      use_instance_pseduo_losses=False,
                      **kwargs,):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #print('gt_bboxes in maskrcnn resnet: ', gt_bboxes)
        x = self.extract_feat(img, use_seg_loss=self.use_seg_loss)
        #breakpoint()
        if self.use_seg_loss:
            x_b, x, score_map = x
        #breakpoint()
        losses = dict()
        if gt_semantic_seg is not None:
            #breakpoint()
            loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_decode)
        
        #results['sem_results'] = seg_pred

        
        if gt_bboxes:
            batch_size = len(gt_labels)
            set_loss_to_zero = False
            for i in range(batch_size):
                if gt_labels[i].numel() == 0:
                    set_loss_to_zero = True
                    
            if not set_loss_to_zero:
                # RPN forward and loss
                if self.with_rpn:
                    proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                    self.test_cfg.rpn)
                    rpn_losses, proposal_list = self.rpn_head.forward_train(
                        # x,
                        # img_metas,
                        # gt_bboxes,
                        # gt_labels=None,
                        # gt_bboxes_ignore=gt_bboxes_ignore,
                        # proposal_cfg=proposal_cfg,
                        # **kwargs)
                        x,
                        img_metas,
                        gt_bboxes,
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg,
                        gt_box_domain_indicator=box_domain_indicator,
                        pseudo_wght_val=pseudo_wght_val,
                        use_instance_pseduo_losses=use_instance_pseduo_losses,
                        **kwargs)
                    losses.update(rpn_losses)
                else:
                    proposal_list = proposals

                roi_losses,bb = self.roi_head.forward_train(
                                                        img,x, img_metas, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                #breakpoint()
                losses.update(roi_losses)
                if self.use_seg_loss:
                    losses.update(self.compute_seg_loss(img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels))

                import copy
                rpn_img=copy.deepcopy(img)
                roi_img=copy.deepcopy(img)
                gt_img=copy.deepcopy(img)
                col=0.0
                from torchvision.utils import save_image
                
                #name=filename+'_original_train'+'.png'
                #breakpoint()


                name=str(self.count)+'_orig_train.png'
                #breakpoint()
                if(self.count%self.patt == 0):
                    save_image((img[0]+2)/2, 'PATH_TO_FILL/'+self.folder+'_train/'+name)
                #breakpoint()
                for i in range(proposal_list[0].shape[0]):
                    col =1
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br,_ = proposal_list[0][i]
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    rpn_img[0,:,x_tl:x_tl + 2, y_tl:y_br] = col
                    rpn_img[0,:,x_br:x_br + 2, y_tl:y_br] = col
                    rpn_img[0,:,x_tl:x_br, y_tl:y_tl + 2] = col
                    rpn_img[0,:,x_tl:x_br, y_br:y_br + 2] = col
                
                for i in range(gt_bboxes[0].shape[0]):
                    col =1
                    y_tl, x_tl, y_br, x_br = gt_bboxes[0][i]#proposal_list[0][i]
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    gt_img[0,:,x_tl:x_tl + 2, y_tl:y_br] = col
                    gt_img[0,:,x_br:x_br + 2, y_tl:y_br] = col
                    gt_img[0,:,x_tl:x_br, y_tl:y_tl + 2] = col
                    gt_img[0,:,x_tl:x_br, y_br:y_br + 2] = col

                #breakpoint()
                for i in range(bb.shape[0]):
                    col =1
                    y_tl, x_tl, y_br, x_br,_ = bb[i]#proposal_list[0][i]
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    roi_img[0,:,x_tl:x_tl + 2, y_tl:y_br] = col
                    roi_img[0,:,x_br:x_br + 2, y_tl:y_br] = col
                    roi_img[0,:,x_tl:x_br, y_tl:y_tl + 2] = col
                    roi_img[0,:,x_tl:x_br, y_br:y_br + 2] = col
                #new_img[:,:,0:1000, 0:2000] = 0
                #breakpoint()
                #print(new_img)
                name=str(self.count)+'_regional_proposals_train.png'
                
                
                from torchvision.utils import save_image
                if(self.count%self.patt == 0):
                    save_image((rpn_img[0]+2.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)

                name=str(self.count)+'_gt_bb_train.png'
                if(self.count%self.patt == 0):
                    save_image((gt_img[0]+2.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)


                name=str(self.count)+'_roi_train.png'
                if(self.count%self.patt == 0):
                    save_image((roi_img[0]+2.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)

                self.count+=1
        return losses
    
    def train_step(self, data, optimizer):
        #breakpoint()
        optimizer.zero_grad()
        losses = self(**data)
        #print(losses)
        if('loss_rpn_cls' in losses):
            
            #print('losses : ', losses)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
            # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
            # outputs = dict(log_vars=log_vars, num_samples=len(data['img_metas']))
            loss.backward()
            optimizer.step()
        else:
            outputs={}

        return outputs

    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels):
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_masks, gt_labels)
        loss = F.binary_cross_entropy(F.sigmoid(score_map), target, weight=mask, reduction='sum')
        loss = loss / mask.sum()
        loss = {'loss_aux_seg': loss}
        return loss

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_masks, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.class_names), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.class_names), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    #x should stay not x_n
    def simple_test(self, img, img_meta, rescale=True, proposals=None):
        #print('##########################################################################')
        #print('########################in forward test of mask###########################')
        #print(img_meta[0]['filename'])
        #breakpoint()
        
        filename = img_meta[0]['filename'].split('/')[-1].split('.')[0]
        #print('-------------- filename:', filename)
        results = {}
        # def inference() part
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        # def encode_decode() part
        #if self.use_neck_feat_for_decode_head:
        #    assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        #print('size of the image is: ', img.size())
        #breakpoint()
        #torch.Size([1, 3, 512, 1024])
        x = self.extract_feat(img,use_seg_loss=True)
        #breakpoint()

        #if self.use_seg_loss:
        x_b, x, score_map = x
        #x = x_n if self.use_neck_feat_for_decode_head else x_b
        #print('length of image: ', len(img), img[0].shape)
        #print('the shape of x_b: ', len(x_b), x_b[0].shape, x_b[1].shape, x_b[2].shape, x_b[3].shape)
        #print('the shape of x_n: ', len(x_n), x_n[0].shape, x_n[1].shape, x_n[2].shape, x_n[3].shape, x_n[4].shape)
        #breakpoint()
        #out_decode_head = self._decode_head_forward_test(x, img_meta)
        #seg_logit = resize(input=out_decode_head, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners) # resizing the predicted semantic map to input image shape
        # def whole_inference() part

        out_decode_head = self._decode_head_forward_test(x_b, img_meta)
        #breakpoint()
        seg_logit = resize(input=out_decode_head, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners) # resizing the predicted semantic map to input image shape
        # def whole_inference() part
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False) # resizing the predicted semantic map to original image shape
        # def inference() part
        #seg_logit.shape torch.Size([1, 19, 1024, 2048])
        output = F.softmax(seg_logit, dim=1)
        #output torch.Size([1, 19, 1024, 2048])
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
        # def simple_test() part
        seg_pred = output.argmax(dim=1)
        #breakpoint()
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        results['sem_results'] = seg_pred
        breakpoint()
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            #seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False) # resizing the predicted semantic map to original image shape
        # def inference() part
        #output = F.softmax(seg_logit, dim=1)
        #flip = img_meta[0]['flip']
        #if flip:
        #    flip_direction = img_meta[0]['flip_direction']
        #    assert flip_direction in ['horizontal', 'vertical']
        #    if flip_direction == 'horizontal':
        #        output = output.flip(dims=(3,))
        #    elif flip_direction == 'vertical':
        #        output = output.flip(dims=(2,))
        # def simple_test() part
        #seg_pred = output.argmax(dim=1)
        #if torch.onnx.is_in_onnx_export():
        #    # our inference backend only support 4D output
        #    seg_pred = seg_pred.unsqueeze(0)
        #    return seg_pred
        #seg_pred = seg_pred.cpu().numpy()
        ## unravel batch dim
        #seg_pred = list(seg_pred)
        #print('seg_pred: ', len(seg_pred))
        #print('seg_pred: ', seg_pred[0].shape)
        #results['sem_results'] = seg_pred
        # maskrcnn stuff below:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
        else:
            proposal_list = proposals
        #breakpoint()
        import copy
        rpn_img=copy.deepcopy(img)
        roi_img=copy.deepcopy(img)
        col=0.0
        #from torchvision.utils import save_image
        from torchvision.utils import save_image
        name=filename+'_orig.png'
        if(self.test_count%1==0):
            save_image((img+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)
        #save_image(img[0], name)
        for i in range(proposal_list[0].shape[0]):
            col =100
            y_tl, x_tl, y_br, x_br,_ = proposal_list[0][i]
            x_tl = int(x_tl)
            y_tl = int(y_tl)
            x_br = int(x_br)
            y_br = int(y_br)
            #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
            #print('imssssss: ', img)
            rpn_img[:,:,x_tl:x_tl + 2, y_tl:y_br] = col
            rpn_img[:,:,x_br:x_br + 2, y_tl:y_br] = col
            rpn_img[:,:,x_tl:x_br, y_tl:y_tl + 2] = col
            rpn_img[:,:,x_tl:x_br, y_br:y_br + 2] = col
        #new_img[:,:,0:1000, 0:2000] = 0
        #print(new_img)
        from torchvision.utils import save_image
        name=filename+'_rpn.png'
        print(filename)
        if(self.test_count%1==0):
            save_image((rpn_img+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)
        #from torchvision.utils import save_image
        #save_image(rpn_img, name)
        
        #save_image(roi_img, 'try.png')
        #import torch as torch
        
        #save_image(new_img, 'img1.png')
        #torch.save(new_img, "faces.png")
        #print('img: ',img)
        #print('torch max: ', torch.max(img))
        #print('torch min: ', torch.min(img))
        #print('img: ',img)
        #print('img_meta: ',  len(img_meta))
        #print('img_meta: ',  img_meta[0])
        #print('image size: ',  img[0].size())
        #print('proposal: ', proposal_list[0].shape)
        #print('proposal_list: ', len(proposal_list))
        #print('proposal: ', proposal_list[0])
        #print(proposal_list[0])
        #inst_pred, inst_pred_feat
        self.takeFeat=False#self.test_cfg['take']#False
        inst_pred_feat=None

        inst_pred,_ = self.roi_head.simple_test(x, proposal_list, img_meta, rescale=rescale)
        #breakpoint()
        num_classes = len(inst_pred[0][0])
        for i in range(num_classes):
            for j in range(inst_pred[0][0][i].shape[0]):
                y_tl, x_tl, y_br, x_br,_ = inst_pred[0][0][i][j]
                #breakpoint()
                x_tl = int(img_meta[0]['scale_factor'][0]*x_tl)
                y_tl = int(img_meta[0]['scale_factor'][1]*y_tl)
                x_br = int(img_meta[0]['scale_factor'][2]*x_br)
                y_br = int(img_meta[0]['scale_factor'][3]*y_br)
                #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                #print('imssssss: ', img)
                roi_img[:,:,x_tl:x_tl + 2, y_tl:y_br] = col
                roi_img[:,:,x_br:x_br + 2, y_tl:y_br] = col
                roi_img[:,:,x_tl:x_br, y_tl:y_tl + 2] = col
                roi_img[:,:,x_tl:x_br, y_br:y_br + 2] = col
        #from torchvision.utils import save_image
        from torchvision.utils import save_image
        name=filename+'_roi.png'
        if(self.test_count%1==0):
            save_image((roi_img+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)
        import numpy as np
        boxes=inst_pred[0][0]
        masks=inst_pred[0][1]
        thing_list_mapids = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
        #breakpoint()
        assert len(boxes) == len(masks), 'boxes and masks lists must have same length!'
        num_classes = len(boxes)
        instances = []
        #breakpoint()
        #print('pred shape: ', pred_shape)
        pred_shape=(1024,2048)
        ins_seg = np.zeros(pred_shape).astype(int)
        #Elham
        ins_seg_cars = np.zeros(pred_shape).astype(int)
        #ins_seg_mask_feat=[[] for i in range(num_classes)]
        ins_seg_mask_cars_feat=[]
        num_cars=0
        color=100
        ins_cars = []
        boxes_cars=[]
        ############
        ins_cnt = 1
        
        mask_score_th=0.5
        for c in range(num_classes):
            boxes_c = boxes[c]
            if boxes_c.shape[0] > 0:
                masks_c = masks[c]
                #breakpoint()

                assert boxes_c.shape[0] == len(masks_c), 'there must be same number of masks and boxes for a class!'
                for m in range(len(masks_c)):
                    mask_score = boxes_c[m, 4]
                    #ins = OrderedDict()
                    #ins['pred_class'] = thing_list_mapids[c]
                    #ins['pred_mask'] = np.array(masks_c[m], dtype='uint8')
                    #ins['score'] = mask_score
                    #instances.append(ins)

                    if mask_score >= mask_score_th:
                        #breakpoint()
                        ins_seg[masks_c[m]] = ins_cnt
                        #ins_seg_mask_feat[c].append(masks_feat_c[m])
                        ins_cnt += 1
                        

                        if(c==2):
                        #breakpoint()
                            num_cars+=1
                            ins_seg_cars[masks_c[m]] = color
                            ins_cars.append(masks_c[m])
                            color+=20
                            boxes_cars.append(boxes_c[m])

        from torchvision.utils import save_image
        filename = img_meta[0]['filename'].split('/')[-1].split('.')[0]
        name=filename+'_mask'+'.png'
        #breakpoint()
        im=torch.stack( [torch.tensor( ins_seg_cars),  torch.tensor( ins_seg_cars),  torch.tensor( ins_seg_cars)])
        save_image(im.float()*0.001, 'PATH_TO_FILL/'+self.folder+'_test/'+name)  
        #print('after roi inst shape: ', inst_pred[0][0][0].shape)
        #print('inst_pred: ', len(inst_pred))
        #print('inst_pred: ', len(inst_pred[0]))
        self.test_count+=1
        #print(inst_pred)
        results['ins_results'] = inst_pred
        results['ins_results_feat']=inst_pred_feat
        return [results] 

    # #x should stay not x_n
    # def simple_test(self, img, img_metas, proposals=None, rescale=True):
    #     """Test without augmentation."""

    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     if proposals is None:
    #         proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals

    #     return self.roi_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
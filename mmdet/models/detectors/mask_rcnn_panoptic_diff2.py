# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------

import pdb
pdb.set_trace()
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector
from mmdet.ops import resize
from mmseg.core import add_prefix
import torch
import torch.nn.functional as F
import copy
from omegaconf import OmegaConf

from VPD.vpd import UNetWrapper, TextAdapter
import torch.nn as nn

from cityscapesscripts.helpers import labels


from ldm.util import instantiate_from_config

@DETECTORS.register_module()
class MaskRCNNPanopticDiff(TwoStageDetector):

    def __init__(self,
                 backbone,
                 decode_head,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 use_neck_feat_for_decode_head=False,
                 auxiliary_head=None):
        super(MaskRCNNPanopticDiff, self).__init__(
                                                backbone=backbone,
                                                neck=neck,
                                                rpn_head=rpn_head,
                                                roi_head=roi_head,
                                                train_cfg=train_cfg,
                                                test_cfg=test_cfg,
                                                pretrained=pretrained,
                                                init_cfg=init_cfg
                                                )
        self._init_decode_head(decode_head)
        # self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_neck_feat_for_decode_head = use_neck_feat_for_decode_head

        # get the unet ready
        sd_path='PATH_TO_FILL/v1-5-pruned-emaonly.ckpt'
        config = OmegaConf.load('stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
        config.model.params.ckpt_path = f'{sd_path}'
        config.model.params.cond_stage_config.target = 'ldm.modules.encoders.modules.AbstractEncoder'
        gamma_init_value=1e-4

        from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
        textEncoderClip = FrozenCLIPTextEmbedder().to('cuda')
        class_embeds_list = []
        #breakpoint()
        #print('labels.labels: ', labels.labels)
        for i in range(len(labels.labels)):
            if(labels.labels[i].trainId !=  255):
                #print('name: ',labels.labels[i].name)
                #source
                embedding=textEncoderClip(['a photo of a' + labels.labels[i].name] )
                class_embeds_list.append(embedding)
        class_embeddings=torch.cat(class_embeds_list, dim=0).to('cpu')
        del textEncoderClip
        del class_embeds_list
        del self.backbone


        # prepare the unet        
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        #print('self.encoder_vq: ', self.encoder_vq)
        unet_config=dict()
        #del self.backbone
        #print('sd_model: ', sd_model)
        #import copy
        #self.try1 =copy.deepcopy(sd_model.model)
        self.backbone = UNetWrapper(sd_model.model, **unet_config)
        sd_model.model = None
        sd_model.first_stage_model = None
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        
        self.sd_model = sd_model

        # class embeddings & text adapter
        #class_embedding_path='./VPD/segmentation/class_embeddings.pth'
        #class_embeddings = torch.load(class_embedding_path)

        #torch.Size([150, 768])
        self.register_buffer('class_embeddings', class_embeddings)
        text_dim = class_embeddings.size(-1)
        #print('-----------------labels: ',labels.labels[0].name)
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)
        #print('FINISHED THIS')
        #breakpoint()
        dev='cuda'

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.text_adapter.parameters():
            param.requires_grad = False
        for param in self.encoder_vq.parameters():
            param.requires_grad = False
        self.conv128 = nn.Conv2d(320, 64, kernel_size=(1,1), stride=1, padding=1).to(dev)
        self.conv64 = nn.Conv2d(660, 128, kernel_size=(1,1), stride=1, padding=1).to(dev)
        self.conv32 = nn.Conv2d(1300, 320, kernel_size=(1,1), stride=1, padding=1).to(dev)
        self.conv16 = nn.Conv2d(1280, 512, kernel_size=(1,1), stride=1, padding=1).to(dev)

    # init daformer decode head
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, img):
        """Extract features from images."""
        #print("self.try: ", self.try1)
        #print('##########################################ENTERING########################################################3')
        x_neck = None
        #x_backbone = self.backbone(img)
        #print('x_backbone shape: ', x_backbone)
        #if self.with_neck:
        #    x_neck = self.neck(x_backbone)
        #print('self unet: ', self.unet)
        #breakpoint()
        """Extract features from images."""
        #print('##########################################GOING INTO THE FEATURE EXTRACTION PART########################################################3')
        with torch.no_grad():
            latents = self.encoder_vq.encode(img)
            self.encoder_vq.eval()
            #print('##########################################one########################################################3')
            latents = latents.mode().detach()
            #print('##########################################two########################################################3')
            c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma) # NOTE: here the c_crossattn should be expand_dim as latents
            self.text_adapter.eval()
            #print('cross attention: ', c_crossattn.shape)
            #print('##########################################three########################################################3')
            t = torch.ones((img.shape[0],), device=img.device).long()
            self.backbone.eval()
            outs = self.backbone(latents, t, c_crossattn=[c_crossattn])
        #print('outs length: ', len(outs))
        #print('outs in feat: ', outs[0].shape, outs[1].shape, outs[2].shape, outs[3].shape,)

        import torch.nn.functional as F
        
        #print('shape of new: ', new.shape)
        new_outs = []

        #64 to 128
        new = self.conv128(outs[0])
        new = F.upsample(new, scale_factor=2)
        new_outs.append(new)

        #32 to 64
        new = self.conv64(outs[1])
        new = F.upsample(new, scale_factor=2)
        new_outs.append(new)

        #16 to 32
        new = self.conv32(outs[2])
        new = F.upsample(new, scale_factor=2)
        new_outs.append(new)

        #8 to 16
        new = self.conv16(outs[3])
        new = F.upsample(new, scale_factor=2)
        new_outs.append(new)

        x_backbone = new_outs
        if self.with_neck:
            x_neck = self.neck(x_backbone)
        
        #
        #conv2 = conv2.to(dev)
        #device = torch.device(dev) 
        #new_channels = conv2(new)
        #breakpoint()
        # print('shape of new_channel: ', new_channels.shape)

        return x_backbone, x_neck

    def encode_decode(self, img, img_metas):
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        x_b, x_n = self.extract_feat(img)
        x = x_n if self.use_neck_feat_for_decode_head else x_b
        out_decode_head = self._decode_head_forward_test(x, img_metas)
        out_decode_head = resize(input=out_decode_head, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out_decode_head

    def encode_decode_full(self, img, img_metas):
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        x_b, x_n = self.extract_feat(img)
        x = x_n if self.use_neck_feat_for_decode_head else x_b
        out_decode_head = self._decode_head_forward_test(x, img_metas)
        out_decode_head = resize(input=out_decode_head, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # maskrcnn stuff below:
        proposal_list = self.rpn_head.simple_test_rpn(x_n, img_metas)
        out_roi_head = self.roi_head.simple_test(x_n, proposal_list, img_metas, rescale=False, isTrain=True) # calls the StandardRoIHead.simple_test()
        return out_decode_head, out_roi_head

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, seg_weight=None):
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg, seg_weight)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    def forward_dummy(self, img):
        """Dummy forward function."""
        # seg_logit = self.encode_decode(img, None)
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        img_metas = None
        x_b, x_n = self.extract_feat(img)
        x = x_n if self.use_neck_feat_for_decode_head else x_b
        out_decode_head = self._decode_head_forward_test(x, img_metas)
        out_decode_head = resize(input=out_decode_head, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # maskrcnn stuff below:
        out_roi_head = ()
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x_n)
            out_roi_head = out_roi_head + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_n, proposals)
        out_roi_head = out_roi_head + (roi_outs,)
        return out_decode_head, out_roi_head


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
                      **kwargs,
                      ):
        #print('in forward train of mask')
        #print('img_metas: ', img_metas)
        #print('the length of the labels: ', len(gt_semantic_seg))
        #print('the length of the groundtruth labels: ', len(gt_labels))
        #print('gt_semantic_seg: ', gt_semantic_seg[0].shape, torch.unique(gt_semantic_seg[0]))
        #print('gt_labels: ', gt_labels[0].shape)
        #print('gt_labels: ', gt_labels[0])
        #print('gt_labels: ', torch.unique(gt_labels[0]))
        #print('gt_bboxes: ', gt_bboxes)
        #print('gt_masks: ', gt_masks[0].shape)
        #print('#################################################################################')
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        #print('length of image before: ', len(img), img[0].shape, img[1].shape)
        x_b, x_n = self.extract_feat(img)
        #print('length of image: ', len(img), img[0].shape, img[1].shape)
        #print('the shape of x_b: ', len(x_b), x_b[0].shape, x_b[1].shape, x_b[2].shape, x_b[3].shape)
        #print('the shape of x_n: ', len(x_n), x_n[0].shape, x_n[1].shape, x_n[2].shape, x_n[3].shape, x_n[4].shape)
        #length of image:  2
        #the shape of x_b:  4 torch.Size([2, 64, 128, 128])
        #the shape of x_n:  5 torch.Size([2, 256, 128, 128])
        #this is false so the backbone is going to be used
        #print('is it true: ', self.use_neck_feat_for_decode_head)
        x = x_n if self.use_neck_feat_for_decode_head else x_b

        #import clip
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #model, preprocess = clip.load("ViT-B/32", device=device)
        #text1 = clip.tokenize(["building", "car", "sidewalk"]).to(device)
        #text2 = clip.tokenize(["building", "car", "sidewalk"]).to(device)
        #text_features1 = model.encode_text(text1)
        #text_features2 = model.encode_text(text2)
        #text_feats = torch.cat([text_features1.unsqueeze(0),text_features2.unsqueeze(0)], dim=0) 
        #print('text_features.shape: ', text_feats.shape)
        #level5 = copy.copy(x_n[-1])
        #print('level_5 shape: ', level5.shape)


        losses = dict()
        if return_feat:
            losses['features'] = x_b # for feat distance loss we always use the backbone feature
        if gt_semantic_seg is not None:
            loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_decode)
        if gt_bboxes:
            batch_size = len(gt_labels)
            set_loss_to_zero = False
            for i in range(batch_size):
                if gt_labels[i].numel() == 0:
                    set_loss_to_zero = True
                    
            if not set_loss_to_zero:
                # RPN forward and loss
                if self.with_rpn:
                    proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
                    rpn_losses, proposal_list = self.rpn_head.forward_train(
                                                                                x_n,
                                                                                img_metas,
                                                                                gt_bboxes,
                                                                                gt_labels=None,
                                                                                gt_bboxes_ignore=gt_bboxes_ignore,
                                                                                proposal_cfg=proposal_cfg,
                                                                                gt_box_domain_indicator=box_domain_indicator,
                                                                                pseudo_wght_val=pseudo_wght_val,
                                                                                use_instance_pseduo_losses=use_instance_pseduo_losses,
                                                                                **kwargs
                                                                            )
                    losses.update(add_prefix(rpn_losses, 'rpn'))
                else:
                    proposal_list = proposals
                # RoIHead (includes box and mask head) forward and loss
                roi_losses = self.roi_head.forward_train(
                                                                x_n,
                                                                img_metas,
                                                                proposal_list,
                                                                gt_bboxes,
                                                                gt_labels,
                                                                gt_bboxes_ignore,
                                                                gt_masks,
                                                                gt_box_domain_indicator=box_domain_indicator,
                                                                pseudo_wght_val=pseudo_wght_val,
                                                                activate_visual_debug=activate_visual_debug,
                                                                use_instance_pseduo_losses=use_instance_pseduo_losses,
                                                                **kwargs
                                                            )
                losses.update(add_prefix(roi_losses, 'roi'))
                # losses.update(roi_losses)
        return losses

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        # outputs = dict(log_vars=log_vars, num_samples=len(data['img_metas']))
        loss.backward()
        optimizer.step()
        return outputs

    # this run inference on both semantic and instance segmentation
    def simple_test(self, img, img_meta, rescale=True, proposals=None):
        #print('in forward test of mask')
        #print(img_meta[0]['filename'])
        filename = img_meta[0]['filename'].split('/')[-1].split('.')[0]
        #print('-------------- filename:', filename)
        results = {}
        # def inference() part
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        # def encode_decode() part
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        #print('size of the image is: ', img.size())
        x_b, x_n = self.extract_feat(img)
        x = x_n if self.use_neck_feat_for_decode_head else x_b
        #print('length of image: ', len(img), img[0].shape)
        #print('the shape of x_b: ', len(x_b), x_b[0].shape, x_b[1].shape, x_b[2].shape, x_b[3].shape)
        #print('the shape of x_n: ', len(x_n), x_n[0].shape, x_n[1].shape, x_n[2].shape, x_n[3].shape, x_n[4].shape)
        out_decode_head = self._decode_head_forward_test(x, img_meta)
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
        output = F.softmax(seg_logit, dim=1)
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
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        #print('seg_pred: ', len(seg_pred))
        #print('seg_pred: ', seg_pred[0].shape)
        results['sem_results'] = seg_pred
        # maskrcnn stuff below:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x_n, img_meta)
        else:
            proposal_list = proposals
        
        rpn_img=copy.deepcopy(img)
        roi_img=copy.deepcopy(img)
        col=0.0
        from torchvision.utils import save_image
        name=filename+'_original'+'.png'
        save_image(img[0], name)
        for i in range(1000):
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
        name=filename+'_regional_proposals.png'
        from torchvision.utils import save_image
        save_image(rpn_img, name)
        
        save_image(roi_img, 'try.png')
        #import torch as torch
        
        #save_image(new_img, 'img1.png')
        #torch.save(new_img, "faces.png")
        #ßprint('img: ',img)
        #print('torch max: ', torch.max(img))
        #print('torch min: ', torch.min(img))
        #print('img: ',img)
        #print('img_meta: ',  len(img_meta))
        #print('img_meta: ',  img_meta[0])
        #print('image size: ',  img[0].size())
        #print('proposal_list: ', len(proposal_list))
        #print('proposal: ', proposal_list[0].shape)
        #print('proposal: ', proposal_list[0])
        #print(proposal_list[0])
        inst_pred = self.roi_head.simple_test(x_n, proposal_list, img_meta, rescale=rescale)
        num_classes = len(inst_pred[0][0])
        for i in range(num_classes):
            for j in range(inst_pred[0][0][i].shape[0]):
                y_tl, x_tl, y_br, x_br,_ = inst_pred[0][0][i][j]
                x_tl = int(x_tl)
                y_tl = int(y_tl)
                x_br = int(x_br)
                y_br = int(y_br)
                #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                #print('imssssss: ', img)
                roi_img[:,:,x_tl:x_tl + 2, y_tl:y_br] = col
                roi_img[:,:,x_br:x_br + 2, y_tl:y_br] = col
                roi_img[:,:,x_tl:x_br, y_tl:y_tl + 2] = col
                roi_img[:,:,x_tl:x_br, y_br:y_br + 2] = col
        from torchvision.utils import save_image
        name=filename+'_roi.png'
        save_image(roi_img, name)


        #print('after roi inst shape: ', inst_pred[0][0][0].shape)
        #print('inst_pred: ', len(inst_pred))
        #print('inst_pred: ', len(inst_pred[0]))

        #print(inst_pred)
        results['ins_results'] = inst_pred
        return [results]

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        return output

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
        return seg_logit


    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def aug_test(self, imgs, img_metas, rescale=True):
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
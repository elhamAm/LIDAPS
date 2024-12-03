# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# Adapted from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for panoptic segmentation
# ------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base_panoptic import BaseSegmentorPanoptic
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder
from VPD.vpd.models import FrozenCLIPEmbedder
from VPD.vpd import UNetWrapper, TextAdapter
from  tqdm import tqdm
import torch.nn as nn
import wandb
import os

from cityscapesscripts.helpers import labels


@SEGMENTORS.register_module()
class EncoderDecoderPanopticDiffFrozenEnc(BaseSegmentorPanoptic):
    """Encoder Decoder segmentors.

    EncoderDecoderPanoptic typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 # eval_type,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 activate_panoptic=False,
                 ):
        super(EncoderDecoderPanopticDiffFrozenEnc, self).__init__(init_cfg)
        print('is this ever called?')
        # self.eval_type = eval_type
        self.activate_panoptic = activate_panoptic

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        #breakpoint()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head
        print(os.getenv('TRANSFORMERS_CACHE'))
        #breakpoint()
        #breakpoint()

        def load_model_from_config(config, ckpt, verbose=False):
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
            #breakpoint()
            model = instantiate_from_config(config.model)
            #breakpoint()
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
            #model.cuda()
            #model.eval()
            return model
            #breakpoint()
        sd_path='PATH_TO_FILL/v1-5-pruned.ckpt'
        config = OmegaConf.load('PATH_TO_FILL/my_edaps/stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
        #breakpoint()
        model = load_model_from_config(config, sd_path)
        #breakpoint()
        del self.backbone
        self.backbone=None

        
        self.unet = UNetWrapper(model.model)#, **unet_config)
        self.unet.to('cuda')   
        self.encoder_vq = model.first_stage_model
        self.encoder_vq.to('cuda') 
        
        #del model.model
        #model.model=None
        #breakpoint()

        text_encoder = FrozenCLIPEmbedder(max_length=20)
        text_encoder.cuda()

        imagenet_templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
        ]
        with torch.no_grad():
            zeroshot_weights = []
            text_encoder.eval()
            all=[]
            for i in range(len(labels.labels)):
                if(labels.labels[i].trainId !=  255 and labels.labels[i].trainId !=  -1):
                    texts=[]
                    #print(labels.labels[i].name)
                    texts = texts + [template.format(labels.labels[i].name) for template in imagenet_templates] #['a photo of a ' + labels.labels[i].name] # #format with class
                    embeddings = text_encoder.encode(texts).detach().mean(dim=0)
                    all.append('a photo of a ' + labels.labels[i].name)
                    #breakpoint()
                    zeroshot_weights.append(embeddings)
        #texts=[]
        #texts = texts + [""] #['a photo of a ' + labels.labels[i].name] # #format with class
        #embeddings = text_encoder.encode(texts).detach().mean(dim=0)
        #zeroshot_weights.append(embeddings)
        #breakpoint()
        #other_embedding=model.get_learned_conditioning(all)
        #print('other_embedding shape: ', other_embedding.shape)
        class_embeddings = torch.stack(zeroshot_weights, dim=0) #other_embedding #
        text_dim = class_embeddings.size(-1)
        gamma_init_value=1e-4
        self.gamma = nn.Parameter(torch.ones(text_dim) * gamma_init_value)
        self.text_adapter = TextAdapter(text_dim=text_dim)
        #breakpoint()

        del text_encoder
        text_encoder=None
        
        
        #sd_model.model = None
        #sd_model.first_stage_model = None
        #del sd_model.cond_stage_model
        #del self.encoder_vq.decoder
        self.register_buffer('class_embeddings', class_embeddings)
        #text_dim = class_embeddings.size(-1)
        #breakpoint()
        print('####################################################')
        print('FINISHED THIS')
        print('####################################################')
        #breakpoint()

        if self.train_cfg is not None:
            self.folder='bottom-up'#self.train_cfg['folder']#'diff_roi_hyperbumble_t0_testHypers_RoiScale_where_scale56_bothWeighted0'
        else:
            self.folder='bottom-up'#self.test_cfg['folder']
        os.makedirs('PATH_TO_FILL/'+self.folder+'_train', exist_ok=True)
        os.makedirs('PATH_TO_FILL/'+self.folder+'_test', exist_ok=True)

        torch.cuda.empty_cache()

        #breakpoint()
        #breakpoint()
        #for param in self.unet.parameters():
        #    param.requires_grad = False
        #for param in self.text_adapter.parameters():
        #    param.requires_grad = False
        #for param in self.encoder_vq.parameters():
        #    param.requires_grad = False
        #for param in self.neck.parameters():
        #    param.requires_grad = False
        #for param in self.rpn_head.parameters():
        #    param.requires_grad = False
        #print('####parameters')

    

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        #breakpoint()
        self.encoder_vq.eval()
        with torch.no_grad():
            latents = self.encoder_vq.encode(img)
            latents = latents.mode().detach()
            latents=latents*0.18215
        bs = latents.shape[0]
        c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma)
        t = torch.zeros((img.shape[0],), device=img.device).long()
        isEval=img.shape[2]!=img.shape[3]
        
        x = self.unet(isEval,latents, t, c_crossattn=[c_crossattn])
        #x = self.unet(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        if not self.activate_panoptic:
            out_semantic = self._decode_head_forward_test(x, img_metas)
            out_semantic = resize(input=out_semantic, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            return out_semantic
        else:
            out_semantic, out_center, out_offset = self._decode_head_forward_test(x, img_metas)
            out_semantic = resize(input=out_semantic, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            out_center = resize(input=out_center, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            out_offset = resize(input=out_offset, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            return out_semantic, out_center, out_offset


    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   gt_center,
                                   center_weights,
                                   gt_offset,
                                   offset_weights,
                                   gt_instance_seg,
                                   gt_depth_map,
                                   seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head.forward_train(x,
                                                     img_metas,
                                                     gt_semantic_seg,
                                                     gt_center,
                                                     center_weights,
                                                     gt_offset,
                                                     offset_weights,
                                                     gt_instance_seg,
                                                     gt_depth_map,
                                                     self.train_cfg,
                                                     seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        #
        if not self.activate_panoptic:
            semantic_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
            return semantic_logits
        else:
            semantic_logits, center_logits, offset_logits, depth_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
            return semantic_logits, center_logits, offset_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      gt_center,
                                      center_weights,
                                      gt_offset,
                                      offset_weights,
                                      gt_instance_seg,
                                      gt_depth_map,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x,
                                                  img_metas,
                                                  gt_semantic_seg,
                                                  gt_center,
                                                  center_weights,
                                                  gt_offset,
                                                  offset_weights,
                                                  gt_instance_seg,
                                                  gt_depth_map,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train( x,
                                                            img_metas,
                                                            gt_semantic_seg,
                                                            gt_center,
                                                            center_weights,
                                                            gt_offset,
                                                            offset_weights,
                                                            gt_instance_seg,
                                                            gt_depth_map,
                                                            self.train_cfg
                                                          )
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      gt_center,
                      center_weights,
                      gt_offset,
                      offset_weights,
                      gt_instance_seg,
                      gt_depth_map,
                      seg_weight=None,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        #breakpoint()
        x = self.extract_feat(img)

        # (Pdb) x[0].shape
        # torch.Size([2, 64, 128, 128])
        # (Pdb) x[1].shape
        # torch.Size([2, 128, 64, 64])
        # (Pdb) x[2].shape
        # torch.Size([2, 320, 32, 32])
        # (Pdb) x[3].shape
        # torch.Size([2, 512, 16, 16])
        #breakpoint()
        import torch.nn.functional as Fun
        x[0] = Fun.interpolate(x[0], [x[0].shape[2]*2, x[0].shape[3]*2], mode='bilinear', align_corners=True)
        x[1] = Fun.interpolate(x[1], [x[1].shape[2]*2, x[1].shape[3]*2], mode='bilinear', align_corners=True)
        x[2] = Fun.interpolate(x[2], [x[2].shape[2]*2, x[2].shape[3]*2], mode='bilinear', align_corners=True)
        x[3] = Fun.interpolate(x[3], [x[3].shape[2]*2, x[3].shape[3]*2], mode='bilinear', align_corners=True)
        losses = dict()
        if return_feat:
            losses['features'] = x

        loss_decode = self._decode_head_forward_train(x,
                                                      img_metas,
                                                      gt_semantic_seg,
                                                      gt_center,
                                                      center_weights,
                                                      gt_offset,
                                                      offset_weights,
                                                      gt_instance_seg,
                                                      gt_depth_map,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                                                            x,
                                                            img_metas,
                                                            gt_semantic_seg,
                                                            gt_center,
                                                            center_weights,
                                                            gt_offset,
                                                            offset_weights,
                                                            gt_instance_seg,
                                                            gt_depth_map,
                                                            seg_weight)
            losses.update(loss_aux)

        return losses

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

    def whole_inference(self, img, img_meta, rescale, debug, eval_type, dataset_name):
        """Inference with full image."""

        if not self.activate_panoptic:
            semantic_logit = self.encode_decode(img, img_meta)
        else:
            semantic_logit, center_logit, offset_logit = self.encode_decode(img, img_meta)

        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                if debug and eval_type == 'panop_deeplab':
                    size = img_meta[0]['img_shape'][:2]
                elif eval_type == 'panop_deeplab':
                    size = img_meta[0]['ori_shape'][:2]
                else:
                    size = img_meta[0]['ori_shape'][:2]

            if not self.activate_panoptic:
                semantic_logit = resize(semantic_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
                return semantic_logit
            else:
                semantic_logit = resize(semantic_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
                center_logit = resize(center_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
                offset_logit = resize(offset_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)

        return semantic_logit, center_logit, offset_logit


    def inference(self, img, img_meta, rescale, eval_type, debug, dataset_name):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        if self.test_cfg.mode == 'slide':
            semantic_logit = self.slide_inference(img, img_meta, rescale)
        else:
            if not self.activate_panoptic:
                semantic_logit = self.whole_inference(img, img_meta, rescale, debug, eval_type, dataset_name)
            else:
                semantic_logit, center_logit, offset_logit = self.whole_inference(img, img_meta, rescale, debug, eval_type, dataset_name)

        if eval_type == 'daformer':
            semantic_logit = F.softmax(semantic_logit, dim=1)  # TODO:

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                semantic_logit = semantic_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                semantic_logit = semantic_logit.flip(dims=(2, ))

        if not self.activate_panoptic:
            return semantic_logit
        else:
            return semantic_logit, center_logit, offset_logit

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        dataset_name = kwargs['eval_kwargs']['dataset_name']
        """Simple test with single image."""
        self.unet.eval()
        self.encoder_vq.eval()
        self.text_adapter.eval()
        
        eval_type = kwargs['eval_kwargs']['eval_type']
        debug = kwargs['eval_kwargs']['debug']

        if not self.activate_panoptic:
            semantic_logit = self.inference(img, img_meta, rescale, eval_type, debug, dataset_name)
        else:
            semantic_logit, center_logit, offset_logit = self.inference(img, img_meta, rescale, eval_type, debug, dataset_name)

        if eval_type == 'daformer':
            semantic_logit = semantic_logit.argmax(dim=1)

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            semantic_logit = semantic_logit.unsqueeze(0)
            return semantic_logit

        if eval_type == 'daformer':
            semantic_logit = semantic_logit.cpu().numpy()
            predictions = list(semantic_logit)
            return predictions

        pred = {}
        if not self.activate_panoptic:
            pred['semantic'] = semantic_logit.cpu().numpy()
        else:
            pred['semantic'] = semantic_logit.cpu().numpy()
            pred['center'] = center_logit.cpu().numpy()
            pred['offset'] = offset_logit.cpu().numpy()
        return pred


    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
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
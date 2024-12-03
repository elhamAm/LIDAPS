# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------

#import pdb
#pdb.set_trace()
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector
from mmdet.ops import resize
from mmseg.core import add_prefix
import torch
import torch.nn.functional as F
import copy
from omegaconf import OmegaConf
from einops import rearrange, repeat

from VPD.vpd.models import FrozenCLIPEmbedder

from VPD.vpd import UNetWrapper, TextAdapter
from  tqdm import tqdm
import torch.nn as nn
import wandb
import os

import torch.nn.functional as F
from cityscapesscripts.helpers import labels


from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenCLIPTextEmbedder

@DETECTORS.register_module()
class MaskRCNNPanopticDiffUnfrozenT0Reverse(TwoStageDetector):

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
        super(MaskRCNNPanopticDiffUnfrozenT0Reverse, self).__init__(
                                                backbone=backbone,
                                                neck=neck,
                                                rpn_head=rpn_head,
                                                roi_head=roi_head,
                                                train_cfg=train_cfg,
                                                test_cfg=test_cfg,
                                                pretrained=pretrained,
                                                init_cfg=init_cfg
                                                )
        
        self.count=0
        self.test_count=0
        #self.beat=True
        self.initialBackBone=False
        self._init_decode_head(decode_head)
        # self._init_auxiliary_head(auxiliary_head)
        print('########################## In the unfrozen version ##############################')
        self.train_cfg = train_cfg
        #breakpoint()
        self.test_cfg = test_cfg
        self.use_neck_feat_for_decode_head = use_neck_feat_for_decode_head
        def load_model_from_config(config, ckpt, verbose=False):
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cpu")
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = pl_sd["state_dict"]
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

            model.cuda()
            #model.eval()
            return model
        if(not self.initialBackBone):
            # get the unet ready
            sd_path='PATH_TO_FILL/v1-5-pruned.ckpt'
            config = OmegaConf.load('stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
            #breakpoint()
            model = load_model_from_config(config, sd_path)
            #breakpoint()
            del self.backbone
            self.backbone=None

            self.unet = UNetWrapper(model.model, use_attn=False)#, **unet_config)   
            self.encoder_vq = model.first_stage_model  

            del model.model
            model.model=None
            #sd_model = instantiate_from_config(config.model)
            #pl_sd = torch.load(sd_path, map_location="cpu")
            #sd = pl_sd["state_dict"]
            #m, u = sd_model.load_state_dict(sd, strict=False)
            #verbose=True
            #if len(m) > 0 and verbose:
            #    print("missing keys:")
            #    print(m)
            #if len(u) > 0 and verbose:
            #    print("unexpected keys:")
            #    print(u)
            #breakpoint()
            
            #unet_config=dict()
            
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
            texts=[]
            texts = texts + [""] #['a photo of a ' + labels.labels[i].name] # #format with class
            #embeddings = text_encoder.encode(texts).detach().mean(dim=0)
            zeroshot_weights.append(embeddings)
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
            print('####################################################')
            print('FINISHED THIS')
            print('####################################################')
            #breakpoint()

            self.folder='diffusion_roi_hyperbumble_t0_testHypers_noattn_rpn'
            
            
            os.makedirs('PATH_TO_FILL/'+self.folder+'_train', exist_ok=True)
            os.makedirs('PATH_TO_FILL/'+self.folder+'_test', exist_ok=True)

            #breakpoint()
            #for param in self.backbone.parameters():
            #    param.requires_grad = False
            #for param in self.text_adapter.parameters():
            #    param.requires_grad = False
            #for param in self.encoder_vq.parameters():
            #    param.requires_grad = False
            #print('####parameters')

            

    # init daformer decode head
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, img):
        """Extract features from images."""
        #print('##########################################ENTERING########################################################3')
        x_neck = None
        #breakpoint()
        if(not self.initialBackBone):
            """Extract features from images."""
            #print('##########################################GOING INTO THE FEATURE EXTRACTION PART########################################################3')
            with torch.no_grad():
                self.encoder_vq.eval()
                latents = self.encoder_vq.encode(img)
                #print('##########################################one########################################################3')
            latents = latents.mode().detach()
            latents=latents*0.18215
            #print('##########################################two########################################################3')
            #self.text_adapter.eval()
            #el=self.text_adapter(latents, self.class_embeddings, self.gamma) 
            #self.class_embeddings.unsqueeze(0)#
            bs = latents.shape[0]
            #c_crossattn = self.class_embeddings #self.text_adapter(latents, self.class_embeddings, self.gamma) #torch.cat([self.class_embeddings.unsqueeze(0),self.class_embeddings.unsqueeze(0)],dim=0)## NOTE: here the c_crossattn should be expand_dim as latents
            #c_crossattn = repeat(self.class_embeddings, 'n c -> b n c', b=bs)
            c_crossattn = self.text_adapter(latents, self.class_embeddings, self.gamma)
            #breakpoint()

            t = torch.zeros((img.shape[0],), device=img.device).long()
            #self.backbone.eval()
            isEval=img.shape[2]!=img.shape[3]
            
            outs = self.unet(isEval,latents, t, c_crossattn=[c_crossattn])
            outs_new = [outs[3],outs[2],outs[1],outs[0]]

            #breakpoint()
            #outs
            #0 #torch.Size([2, 320, 64, 64])
            #1 torch.Size([2, 660, 32, 32])
            #2 torch.Size([2, 1300, 16, 16])
            #3 torch.Size([2, 1280, 8, 8])

            #outs_orig
            #torch.Size([2, 1280, 8, 8])
            #torch.Size([2, 1300, 16, 16])
            #torch.Size([2, 660, 32, 32])
            #torch.Size([2, 320, 64, 64])
            x_backbone = outs_new
        else:
            x_backbone = self.backbone(img)

        #neck
        #0 #torch.Size([2, 256, 8, 8])
        #1 #torch.Size([2, 256, 16, 16])
        #2 #torch.Size([2, 256, 32, 32])
        #3 #torch.Size([2, 256, 64, 64])
        #4 #torch.Size([2, 256, 32, 32])
        if self.with_neck:
            x_neck = self.neck(x_backbone)#outs_orig)
        #breakpoint()

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

        #print('#################################################################################')
        #breakpoint()
        filename = img_metas[0]['filename'].split('/')[-1].split('.')[0]
        #print('filename: ', filename)
        if self.use_neck_feat_for_decode_head:
            assert self.with_neck, 'self.with_neck is False, so set the self.use_neck_feat_for_decode_head to False.'
        #print('length of image before: ', len(img), img[0].shape, img[1].shape)
        x_b, x_n = self.extract_feat(img)
        #print('img: ', img)
        x = x_n if self.use_neck_feat_for_decode_head else x_b


        losses = dict()
        if return_feat:
            losses['features'] = x_b # for feat distance loss we always use the backbone feature
        #breakpoint()
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
                    #breakpoint()
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
                roi_losses, bb = self.roi_head.forward_train(x_n,
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
                #breakpoint()
                
                #breakpoint()
                ###############################################
                rpn_img=copy.deepcopy(img)
                roi_img=copy.deepcopy(img)
                gt_img=copy.deepcopy(img)
                col=0.0
                from torchvision.utils import save_image
                
                #name=filename+'_original_train'+'.png'
                #breakpoint()


                name=str(self.count)+'_orig_train.png'
                if(self.count%200 == 0):
                    save_image((img[0]+1.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)
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
                if(self.count%200 == 0):
                    save_image((rpn_img[0]+1.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)

                name=str(self.count)+'_gt_bb_train.png'
                if(self.count%200 == 0):
                    save_image((gt_img[0]+1.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)


                name=str(self.count)+'_roi_train.png'
                if(self.count%200 == 0):
                    save_image((roi_img[0]+1.)/2., 'PATH_TO_FILL/'+self.folder+'_train/'+name)

                self.count+=1
                # losses.update(roi_losses)
        return losses

    def train_step(self, data, optimizer):
        #breakpoint()
        optimizer.zero_grad()
        losses = self(**data)
        #breakpoint()
        loss, log_vars = self._parse_losses(losses)
        #wandb.log({"loss": loss, 'loss_vars':log_vars,'lr':optimizer.param_groups[-1]['lr']})
        #breakpoint()
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        # outputs = dict(log_vars=log_vars, num_samples=len(data['img_metas']))
        loss.backward()
        optimizer.step()
        return outputs

    # this run inference on both semantic and instance segmentation
    def simple_test(self, img, img_meta, rescale=True, proposals=None):
        #print('are you here??')
        #breakpoint()
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
        self.text_adapter.eval()
        self.unet.eval()
        with torch.no_grad():
            x_b, x_n = self.extract_feat(img)
            x = x_n if self.use_neck_feat_for_decode_head else x_b
        #print('length of image: ', len(img), img[0].shape)
        #print('the shape of x_b: ', len(x_b), x_b[0].shape, x_b[1].shape, x_b[2].shape, x_b[3].shape)
        #print('the shape of x_n: ', len(x_n), x_n[0].shape, x_n[1].shape, x_n[2].shape, x_n[3].shape, x_n[4].shape)
        with torch.no_grad():
            out_decode_head = self._decode_head_forward_test(x, img_meta)
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
        with torch.no_grad():
            seg_pred = output.argmax(dim=1)
        #breakpoint()
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
        with torch.no_grad():
            if proposals is None:
                proposal_list = self.rpn_head.simple_test_rpn(x_n, img_meta)
            else:
                proposal_list = proposals
        ################################# save eval image
        rpn_img=copy.deepcopy(img)
        roi_img=copy.deepcopy(img)
        col=0.0
        from torchvision.utils import save_image
        name=filename+'_original'+'.png'
        if(self.test_count%50==0):
            save_image((img[0]+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)
        #breakpoint()
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
        name=filename+'_regional_proposals.png'
        from torchvision.utils import save_image
        if(self.test_count%50==0):
            save_image((rpn_img+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)
        ################################# save eval image
        # save_image(roi_img, 'try.png')


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
        #breakpoint()
        with torch.no_grad():
            inst_pred,_ = self.roi_head.simple_test(x_n, proposal_list, img_meta, rescale=rescale)
        #breakpoint()
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
        if(self.test_count%50==0):
            save_image((roi_img+1.)/2., 'PATH_TO_FILL/'+self.folder+'_test/'+name)


        #print('after roi inst shape: ', inst_pred[0][0][0].shape)
        #print('inst_pred: ', len(inst_pred))
        #print('inst_pred: ', len(inst_pred[0]))

        #print(inst_pred)
        results['ins_results'] = inst_pred
        return [results]

    def inference(self, img, img_meta, rescale):
        print('#####################')
        print('inference')
        print('#####################')
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
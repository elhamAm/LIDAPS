# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# ---------------------------------------------------------------


import math
import os
import random
from copy import deepcopy
import mmcv
import numpy as np
from mmdet.core import *
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
from mmseg.core import add_prefix
from mmdet.models import UDA, build_detector
from mmdet.models.uda.uda_decorator import UDADecorator, get_module
from mmdet.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform)
from mmseg.utils.visualize_pred  import subplotimg
from mmdet.utils.utils import downscale_label_ratio
import copy

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)
    return norm


@UDA.register_module()
class DACSPSTARGETSCORES(UDADecorator):

    def __init__(self, **cfg):
        super(DACSPSTARGETSCORES, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.share_src_backward = cfg['share_src_backward']
        self.disable_mix_masks = cfg['disable_mix_masks']
        assert self.mix == 'class'
        self.debug_fdist_mask = None
        self.debug_gt_rescale = None
        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        #breakpoint()
        self.ema_model = build_detector(ema_cfg)
        #self.enable_fdist=True
        #breakpoint()
        if self.enable_fdist:
            self.imnet_model = build_detector(deepcopy(cfg['model']))
        else:
            self.imnet_model = None
        #breakpoint()
        self.map_pos_index_to_cid = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
        self.map_cid_to_pos_index = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7}
        self.thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
        self.label_divisor = 1000
        self.label_divisor_target = 10000
        self.iter=0


    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(), self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data +  (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            #breakpoint()
            self.get_imnet_model().eval()
            feat_imnet, _ = self.get_imnet_model().extract_feat(img)
            #breakpoint()
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        #breakpoint()
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor, self.fdist_scale_min_ratio, self.num_classes, 255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses({'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def train_step(self, data_batch, optimizer, **kwargs):
        #breakpoint()
        optimizer.zero_grad()
        #breakpoint()
        log_vars = self(**data_batch)
        optimizer.step()
        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        #breakpoint()
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, # daformer args
                      gt_bboxes, gt_labels, gt_masks,        # maskrcnn args
                      target_img, target_img_metas,
                      gt_panoptic_only_thing_classes,
                      max_inst_per_class,
                      ):        # daformer args
        # [  'gt_masks', 'target_img_metas', 'target_img']
        #breakpoint()
        
        #self.get_imnet_model()=None
        #del self.get_imnet_model()
        #breakpoint()

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        #breakpoint()

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(img,
                                                      img_metas,
                                                      gt_semantic_seg,
                                                      return_feat=True,
                                                      gt_bboxes=gt_bboxes,
                                                      gt_labels=gt_labels,
                                                      gt_masks=gt_masks,
                                                      )

        src_feat = clean_losses.pop('features')
        #breakpoint()
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        if not self.share_src_backward:
            clean_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [ p.grad.detach().clone() for p in params if p.grad is not None ]
                grad_mag = calc_grad_magnitude(seg_grads)
                mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')
        #breakpoint()
        #breakpoint()
        # ImageNet feature distance
        if self.enable_fdist:
            #breakpoint()
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.share_src_backward:
                clean_loss = clean_loss + feat_loss
            else:
                feat_loss.backward()
                if self.print_grad_magnitude:
                    params = self.get_model().backbone.parameters()
                    fd_grads = [ p.grad.detach() for p in params if p.grad is not None ]
                    fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                    grad_mag = calc_grad_magnitude(fd_grads)
                    mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Shared source backward
        if self.share_src_backward:
            clean_loss.backward()
        #torch.cuda.empty_cache()
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        #breakpoint()
        ema_logits = self.get_ema_model().encode_decode(target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        #breakpoint()
        # Apply mixing
        #breakpoint()
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        if self.disable_mix_masks:
            for i in range(batch_size):
                mix_masks[i][:] = 0
                assert mix_masks[i].sum() == 0, 'problem found'

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(strong_parameters, data=torch.stack((img[i], target_img[i])), target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mixed_bboxes, mixed_labels, mixed_masks = None, None, None

        mix_losses = self.get_model().forward_train(mixed_img,
                                                    img_metas,
                                                    mixed_lbl,
                                                    seg_weight=pseudo_weight,
                                                    return_feat=True,
                                                    gt_bboxes=mixed_bboxes,
                                                    gt_labels=mixed_labels,
                                                    gt_masks=mixed_masks,
                                                    )
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        #breakpoint()
        #len 2
        #(Pdb) gt_bboxes[0].shape
        #torch.Size([26, 4])
        #gt_masks[0].masks.shape
        #(26, 512, 512)
        out_roi_head,_ = self.get_ema_model().encode_decode_full(target_img, target_img_metas)
        #boxes = [out_roi_head][0][0], out_roi_head][1][0]]
        #masks = [out_roi_head[0][1], out_roi_head[1][1]]

        targetBoxes=[]
        targetMasks=[]
        targetLabels=[]
        targetScores=[]

        passThis=False
        for j in range(2):
            boxesOneSet=[]
            masksOneSet=[]
            labelsOneSet=[]
            scoresOneSet=[]
            for c in range(8):
                #boxes out_roi_head[j][0]
                #breakpoint()
                nums=out_roi_head[j][0][c][:,:4].shape[0]
                if(nums>0):
                    boxesOneSet.append(torch.tensor(out_roi_head[j][0][c][:,:4]))
                    scoresOneSet.append(torch.tensor(out_roi_head[j][0][c][:,4]))
                    
                    #masks out_roi_head[j][1]

                    masksOneSet.append(torch.tensor(np.array(out_roi_head[j][1][c])))
                    nums=out_roi_head[j][0][c][:,:4].shape[0]
                    labelsOneSet=labelsOneSet+([c] *nums)

            #breakpoint()
            if(len(boxesOneSet)==0):
                passThis=True
                break
            targetBoxes.append(torch.cat(boxesOneSet).to('cuda'))
            targetScores.append(torch.cat(scoresOneSet).to('cuda'))
            #breakpoint()
            targetMasks.append(BitmapMasks(torch.cat(masksOneSet).cpu().numpy(), 512, 512))
            #print('length: ', len(labelsOneSet))
            #print(labelsOneSet)
            
            targetLabels.append(torch.tensor(labelsOneSet).to('cuda'))
        #print('targetBoxes[0].shape: ', targetBoxes[0].shape)
        #print('targetBoxes[1].shape: ', targetBoxes[1].shape)
        #print('labelsOneSet: ', torch.tensor(labelsOneSet).to('cuda'))
        #print('targetLabels: ', targetLabels)
        #breakpoint()
        

        visual=False
        #breakpoint()
        repo='PATH_TO_FILL/images/'
        img_pastes=[]
        if(visual and not passThis):
            img_pastes.append(copy.deepcopy(target_img[0]))
            img_pastes.append(copy.deepcopy(target_img[1]))
            from torchvision.utils import save_image
            
            save_image((target_img[0]+2.)/2., repo+'save0.png')
            save_image((target_img[1]+2.)/2., repo+'save1.png')
            for j in range(batch_size):  
                save_image((img_pastes[j]+2.)/2., './cars_'+str(self.iter)+'_'+str(j)+'.png')
            #breakpoint()
            
            j=0
            for q in range(targetBoxes[j].shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br = targetBoxes[j][q]#proposal_list[0][i]
                    #print('######### y: ',abs(y_tl-y_br), 'x: ', abs(x_tl-x_br))
                    #if(abs(y_tl-y_br) > 85 and abs(x_tl-x_br) > 85):
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    img_pastes[j][:,x_tl:x_tl + 2, y_tl:y_br] = col
                    img_pastes[j][:,x_br:x_br + 2, y_tl:y_br] = col
                    img_pastes[j][:,x_tl:x_br, y_tl:y_tl + 2] = col
                    img_pastes[j][:,x_tl:x_br, y_br:y_br + 2] = col
            save_image((img_pastes[j]+2.)/2., repo+'carsBoxes_'+str(self.iter)+'.png')
            #breakpoint()

            masks=torch.tensor(targetMasks[j].masks).to('cuda')
            #breakpoint()

            
            #inds=gt_labels_inst[]
            for q in range(masks.shape[0]):
                #breakpoint()
                #if(gt_labels_inst[1][q]==2):
                img_pastes[j]=(1-(masks[q]*1).unsqueeze(0).repeat(3,1,1))*img_pastes[j]
            save_image((img_pastes[j]+2)/2., repo+'carsMasks_'+str(self.iter)+'.png')
            self.iter+=1
            #breakpoint()     
        if(not passThis):

            inst_losses = self.get_model().forward_train(target_img,
                                                    target_img_metas,
                                                    gt_semantic_seg=None,
                                                    scores_instance=targetScores,
                                                    return_feat=True,
                                                    gt_bboxes=targetBoxes,
                                                    gt_labels=targetLabels,
                                                    gt_masks=targetMasks,
                                                    )

            inst_losses.pop('features')
            inst_losses = add_prefix(inst_losses, 'inst_mixed')
            inst_loss, inst_log_vars = self._parse_losses(inst_losses)
            log_vars.update(inst_log_vars)
            inst_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots( rows,  cols, figsize=(3 * cols, 3 * rows), gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0 }, )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(axs[0][1], gt_semantic_seg[j], 'Source Seg GT', cmap='cityscapes')
                subplotimg(axs[1][1], pseudo_label[j], 'Target Seg (Pseudo) GT', cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred", cmap="cityscapes")
                subplotimg(axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(axs[0][4], self.debug_fdist_mask[j][0], 'FDist Mask', cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(axs[1][4], self.debug_gt_rescale[j], 'Scaled GT', cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1
        #breakpoint()
        self.iter+=1

        return log_vars

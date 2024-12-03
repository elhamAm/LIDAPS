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
from mmdet.core import *
import mmcv
import numpy as np
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
import os
from torchvision import transforms
from mmseg.utils.visualize_pred import prep_ins_for_vis_mine, prep_ins_for_vis_mine_one


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
class DACSPSOCCPARAMNODELETENEW2(UDADecorator):

    def __init__(self, **cfg):
        super(DACSPSOCCPARAMNODELETENEW2, self).__init__(**cfg)
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

    def runOtherDirection(self, masks_forVis_corrected, conf_forVis_corrected, q_true, thres1, thres2, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas, mixed_img2, mix_masks2):
        out_roi_head,_ = self.get_ema_model().encode_decode_full(target_img, target_img_metas)
        targetBoxes=[]
        targetMasks=[]
        targetLabels=[]

        IMG_MEAN = [123.675/255., 116.28/255., 103.53/255.]
        IMG_STD = [58.395/255., 57.12/255., 57.375/255.]
        def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
            # 3, H, W, B
            #breakpoint()
            x=x.unsqueeze(3)
            ten = x.clone()#.permute(1, 2, 3, 0)
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            #breakpoint()
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2).squeeze(0)

        #gt_bboxes
        #gt_labels
        #gt_masks
        #breakpoint()
        passThis=False
        for j in range(2):
            boxesOneset=[]
            masksOneSet=[]
            labelsOneSet=[]
            for c in range(8):
                #boxes out_roi_head[j][0]
                #breakpoint()
                nums=out_roi_head[j][0][c][:,:4].shape[0]
                if(nums>0):

                    
                    #breakpoint()
                    masks_c = out_roi_head[j][1][c]
                    #print('length of masks: ', len(masks_c))
                    for m in range(len(masks_c)):
                        if(np.unique(masks_c[m]).shape[0]>1):
                            minIndx=np.min(np.argwhere(masks_c[m]>0)[:, 0])
                            maxIndx=np.max(np.argwhere(masks_c[m]>0)[:, 0])
                            minIndy=np.min(np.argwhere(masks_c[m]>0)[:, 1])
                            maxIndy=np.max(np.argwhere(masks_c[m]>0)[:, 1])

                            boxesOneset.append(torch.tensor([minIndy,minIndx, maxIndy, maxIndx ]).unsqueeze(0))
                            #boxesOneset.append(torch.tensor(out_roi_head[j][0][c][:,:4]))
                            
                            #masks out_roi_head[j][1]

                            masksOneSet.append(torch.tensor(np.array(out_roi_head[j][1][c][m])).unsqueeze(0))
                            #breakpoint()
                            #nums=out_roi_head[j][0][c][:,:4].shape[0]
                            labelsOneSet=labelsOneSet+([c] )                   

            #breakpoint()
            #breakpoint()
            if(len(boxesOneset)==0):
                passThis=True
                break
            targetBoxes.append(torch.cat(boxesOneset).to('cuda'))
            targetMasks.append(BitmapMasks(torch.cat(masksOneSet).cpu().numpy(), 512, 512))
            #print('length: ', len(labelsOneSet))
            #breakpoint()
            #print(labelsOneSet)
            
            targetLabels.append(torch.tensor(labelsOneSet).to('cuda'))
            #breakpoint()
        #print('targetBoxes[0].shape: ', targetBoxes[0].shape)
        #print('targetBoxes[1].shape: ', targetBoxes[1].shape)
        #print('labelsOneSet: ', torch.tensor(labelsOneSet).to('cuda'))
        #print('targetLabels: ', targetLabels)
        masks_mixed=[]
        boxes_mixed=[]
        labels_mixed=[]

        mix_instance_img=[]
        for j in range(2):
            mask_target = torch.zeros(target_img[0].shape).to('cuda')
            #for i in range(gt_labels[j].shape[0]):
            #    mask_target[:,gt_masks[j].masks[i]] = 1
            for i in range(masks_forVis_corrected[j].shape[0]):
                if i>=(masks_forVis_corrected[j].shape[0]-q_true[j]):
                    mask_target[:,masks_forVis_corrected[j][i]] = 1
            mix_instance_img.append((mask_target)*img[j]+(1-mask_target)*target_img[j])

        labels_mixed_corrected=[[],[]]
        masks_mixed_corrected=[[],[]]
        boxes_mixed_corrected=[[],[]]

        labels_stacked=[]
        labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))
        labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))

        for j in range(2):
            masks_mixed.append(torch.cat([torch.tensor(targetMasks[j].masks).to('cuda'), torch.tensor(gt_masks[j].masks).to('cuda')]))
            boxes_mixed.append(torch.cat([targetBoxes[j], gt_bboxes[j]]))
            labels_mixed.append(torch.cat([targetLabels[j], gt_labels[j]]))
        #breakpoint()
        if(not passThis):   
            for j in range(2):

                #print('masks_mixed len: ', len(masks_mixed))
                for masksi in range(masks_mixed[j].shape[0]-1,-1,-1):
                    #breakpoint()
                    #labels_stacked_0[masksi]
                    #breakpoint()
                    #print('masksi: ', masksi)
                    if(masksi >= gt_bboxes[j].shape[0]):
                        labels_stacked[j][masks_mixed[j][masksi]==1]+=1
                        labels_stacked[j][labels_stacked[j]>1]=1
                        labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                        masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                        boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                    else:
                        labels_stacked_=copy.deepcopy(labels_stacked)
                        labels_stacked_[j][masks_mixed[j][masksi]==1]+=1
                        #breakpoint()
                        if(not(labels_stacked_[j][labels_stacked_[j]>1].shape[0]>0)):

                            labels_stacked=labels_stacked_
                            #breakpoint()
                            #print('masksi: ', masksi)
                            labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                            masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                            boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                        else:
                            #breakpoint()
                            numbeOfPixels=masks_mixed[j][masksi][masks_mixed[j][masksi]>0].shape[0]
                            numberOfOverlaps=masks_mixed[j][masksi][labels_stacked_[j]>1].shape[0]
                            percentage=numberOfOverlaps/numbeOfPixels*100

                            masks_mixed[j][masksi][labels_stacked_[j]>1]=0
                            #breakpoint()
                            if(torch.unique(masks_mixed[j][masksi]).shape[0]>1):#removed percentage
                                labels_stacked_[j][labels_stacked_[j]>1]=1
                                labels_stacked=labels_stacked_
                                

                                minIndx=np.min(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 0])
                                maxIndx=np.max(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 0])
                                minIndy=np.min(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 1])
                                maxIndy=np.max(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 1])

                                labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                                masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                                boxes_mixed_corrected[j].append(torch.tensor([minIndy,minIndx, maxIndy, maxIndx ]).to('cuda'))

                    
                        
                #breakpoint()  
                masks_mixed_corrected[j]=torch.stack(masks_mixed_corrected[j]).cpu().numpy() 
                masks_mixed_corrected[j]=BitmapMasks(masks_mixed_corrected[j],masks_mixed_corrected[j].shape[1], masks_mixed_corrected[j].shape[2])
                boxes_mixed_corrected[j]=torch.stack(boxes_mixed_corrected[j])
                labels_mixed_corrected[j]=torch.stack(labels_mixed_corrected[j])
                
            #breakpoint()
            #print('delete: ', delete) 
            #breakpoint()

        #gt_masks_inst[0]=BitmapMasks(all_masks0.cpu().numpy(), rolled_masks.shape[1], rolled_masks.shape[2])
                
        

        visual=True
        #breakpoint()
        repo='PATH_TO_FILL/images_source2/'
        img_pastes=[]
        if(visual and not passThis):
            img_pastes.append(copy.deepcopy(mix_instance_img[0]))
            img_pastes.append(copy.deepcopy(mix_instance_img[1]))
            from torchvision.utils import save_image
            
            save_image((img[0]+2.)/2., repo+'source0.png')
            save_image((img[1]+2.)/2., repo+'source1.png')

            save_image((target_img[0]+2.)/2., repo+'target0_'+str(self.iter)+'_.png')
            save_image((target_img[1]+2.)/2., repo+'target1_'+str(self.iter)+'_.png')
            for j in range(batch_size):  
                save_image((img_pastes[j]+2.)/2., repo+'mixed_'+str(self.iter)+'.png')
            #breakpoint()
            
            j=1
            for q in range(boxes_mixed_corrected[j].shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br = boxes_mixed_corrected[j][q]#proposal_list[0][i]
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
            img_pastes_maskOrig=copy.deepcopy(img)
            for q in range(gt_masks[j].masks.shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    #y_tl, x_tl, y_br, x_br = gt_masks[j].masks[q]#proposal_list[0][i]
                    #breakpoint()
                    minIndx=np.min(np.argwhere(gt_masks[j].masks[q]>0)[:, 0])
                    maxIndx=np.max(np.argwhere(gt_masks[j].masks[q]>0)[:, 0])

                    minIndy=np.min(np.argwhere(gt_masks[j].masks[q]>0)[:, 1])
                    maxIndy=np.max(np.argwhere(gt_masks[j].masks[q]>0)[:, 1])

                    #print('######### y: ',abs(y_tl-y_br), 'x: ', abs(x_tl-x_br))
                    #if(abs(y_tl-y_br) > 85 and abs(x_tl-x_br) > 85):
                    x_tl = minIndx
                    y_tl = minIndy
                    x_br = maxIndx
                    y_br = maxIndy
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    img_pastes_maskOrig[j][:,x_tl:x_tl + 2, y_tl:y_br] = col
                    img_pastes_maskOrig[j][:,x_br:x_br + 2, y_tl:y_br] = col
                    img_pastes_maskOrig[j][:,x_tl:x_br, y_tl:y_tl + 2] = col
                    img_pastes_maskOrig[j][:,x_tl:x_br, y_br:y_br + 2] = col
            save_image((img_pastes_maskOrig[j]+2.)/2., repo+'carsMasks_orig_'+str(self.iter)+'.png')

            for q in range(gt_bboxes[j].shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br = gt_bboxes[j][q]#proposal_list[0][i]
                    #print('######### y: ',abs(y_tl-y_br), 'x: ', abs(x_tl-x_br))
                    #if(abs(y_tl-y_br) > 85 and abs(x_tl-x_br) > 85):
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    img[j][:,x_tl:x_tl + 2, y_tl:y_br] = col
                    img[j][:,x_br:x_br + 2, y_tl:y_br] = col
                    img[j][:,x_tl:x_br, y_tl:y_tl + 2] = col
                    img[j][:,x_tl:x_br, y_br:y_br + 2] = col
            save_image((img[j]+2.)/2., repo+'carsBoxes_orig_'+str(self.iter)+'.png')
            #breakpoint()
            #breakpoint()
            masks=torch.tensor(masks_mixed_corrected[j].masks).to('cuda')
            #breakpoint()

            img_insts=[]
            img_insts.append(torch.zeros((512,512)).to('cuda'))
            img_insts.append(torch.zeros((512,512)).to('cuda'))
            #inds=gt_labels_inst[]
            id_inst=1
            for q in range(masks.shape[0]):
                #breakpoint()
                #if(gt_labels_inst[1][q]==2):
                img_pastes[j]=(1-masks[q].unsqueeze(0).repeat(3,1,1))*img_pastes[j]
            oneTime=False
            for q in range(masks_forVis_corrected[j].shape[0]):
                #breakpoint()
                #if(q > gt_bboxes[j].shape[0] and oneTime==False):
                if(conf_forVis_corrected[j][q] > thres2):
                    if q<(masks_forVis_corrected[j].shape[0]-q_true[j]):
                        if(self.iter==1 and q==10):
                            print('none')
                        #    oneTime=True
                        #if(self.iter==2):
                        #    breakpoint()
                        else:#(conf_forVis_corrected[j][q] > thres2):
                            img_insts[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                            id_inst+=1
                    else:
                        img_insts[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                        id_inst+=1
            img_insts[j]=img_insts[j].int()
            save_image((img_pastes[j]+2)/2., repo+'carsMasks_'+str(self.iter)+'.png')
            #self.iter+=1

            img_pass = denormalize(mix_instance_img[j])*255.
            #breakpoint()
            img_new =prep_ins_for_vis_mine_one(img_insts[j].cpu(),  mode='GT', dataset_name='synthia', debug=False, blend_ratio=0.5, img=img_pass.cpu(), show_only_person=False, sem_id=None)
            save_image((img_new)/255., repo+'MasksInst_'+str(self.iter)+'.png')
        #breakpoint()
        #print('here')
        #if(True in
    def runInstances(self,thres1, thres2, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas, mixed_img2, mix_masks2):

        apply=True
        #breakpoint()
        repo='PATH_TO_FILL/images_thres/'
        os.makedirs(repo,exist_ok=True)

        img_pastes=copy.deepcopy(img)
        out_roi_head,_ = self.get_ema_model().encode_decode_full(target_img, target_img_metas)
        boxes_mixed=[]
        masks_mixed=[]
        labels_mixed=[]

        all_masks_forVis_=[]
        all_conf_forVis_=[]

        mix_instance_img=[]
        
        for j in range(2):
            mask_target = torch.zeros(target_img[0].shape).to('cuda')
            boxes = out_roi_head[j][0]
            masks = out_roi_head[j][1]
            
            num_classes=8
            mask_score_th=thres2
            #breakpoint()
            
            #breakpoint()
            passThis =True
            classes=[]
            #breakpoint()
            for c in range(0,8):
                if(boxes[c].shape[0] > 0 and True in list(boxes[c][:,-1]>mask_score_th)):
                    classes.append(c)
                    passThis=False

            if(passThis):
                break

            new_boxes=[]
            new_masks=[]
            new_labels=[]
            all_masks_forVis=[]
            all_conf_forVis=[]
            
            id=0
            #breakpoint()
            if(not passThis):
                for c in classes:
                    boxes_c = boxes[c]
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
                        if np.unique(masks_c[m]).shape[0]>1:
                            all_masks_forVis.append(torch.tensor(masks_c[m]))
                            all_conf_forVis.append(mask_score)

                        if mask_score >= mask_score_th and np.unique(masks_c[m]).shape[0]>1:
                            #breakpoint()
                            mask_target[:,masks_c[m]] = 1

                            #print('1: ', np.unique(masks_c[m]).shape[0])
                            #print('2: ', np.min(np.argwhere(masks_c[m]>0)[:, 0]))
                            #breakpoint()
                            minIndx=np.min(np.argwhere(masks_c[m]>0)[:, 0])
                            maxIndx=np.max(np.argwhere(masks_c[m]>0)[:, 0])
                            minIndy=np.min(np.argwhere(masks_c[m]>0)[:, 1])
                            maxIndy=np.max(np.argwhere(masks_c[m]>0)[:, 1])
                            #breakpoint()
                            new_boxes.append(torch.tensor([minIndy,minIndx, maxIndy, maxIndx ]))

                            new_masks.append(torch.tensor(masks_c[m]))
                            
                            new_labels.append(c)
                    #breakpoint()
                #breakpoint()
                #c is the class number
                id+=1
                mix_instance_img.append(mask_target*target_img[j]+(1-mask_target)*img[j])
                #mix_instance_img.append(mask_target*2*id+(1-mask_target)*img[j])
                #breakpoint()
                if(len(new_boxes) ==0):
                    passThis=True
                    break
                #breakpoint()
                boxes_mixed.append(torch.cat([gt_bboxes[j], torch.stack(new_boxes).to('cuda')]))
                #breakpoint()
                masks_mixed.append(torch.cat([torch.tensor(gt_masks[j].masks).to('cuda'), torch.stack(new_masks).to('cuda')]))
                all_masks_forVis_.append(torch.cat([torch.tensor(gt_masks[j].masks).to('cuda'), torch.stack(all_masks_forVis).to('cuda')]))
                #breakpoint()
                all_conf_forVis_.append(torch.cat([torch.tensor([1.]*gt_labels[j].shape[0]).to('cuda'), torch.tensor(all_conf_forVis).to('cuda')]))
                #labels_mixed.append(torch.cat([gt_labels[j], torch.tensor(new_labels).to('cuda')]))
                
                
                #breakpoint()
                labels_mixed.append(torch.cat([gt_labels[j], torch.tensor(new_labels).to('cuda')]))
                
            #breakpoint()
            #breakpoint()
            labels_mixed_corrected=[[],[]]
            masks_mixed_corrected=[[],[]]
            masks_forVis_corrected=[[],[]]
            conf_forVis_corrected=[[],[]]
            boxes_mixed_corrected=[[],[]]

            labels_stacked=[]
            labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))
            labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))
            #breakpoint()
            #masks01=[]
            #masks01.append(torch.tensor(gt_masks_inst[0].masks).to('cuda'))
            #masks01.append(torch.tensor(gt_masks_inst[1].masks).to('cuda'))
        if(not passThis):   
            for j in range(2):

                #print('masks_mixed len: ', len(masks_mixed))
                for masksi in range(masks_mixed[j].shape[0]-1,-1,-1):
                    #breakpoint()
                    #labels_stacked_0[masksi]
                    #breakpoint()
                    #print('masksi: ', masksi)
                    if(masksi >= gt_bboxes[j].shape[0]):
                        labels_stacked[j][masks_mixed[j][masksi]==1]+=1
                        labels_stacked[j][labels_stacked[j]>1]=1
                        labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                        masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                        boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                    else:
                        labels_stacked_=copy.deepcopy(labels_stacked)
                        labels_stacked_[j][masks_mixed[j][masksi]==1]+=1
                        #breakpoint()
                        if(not(labels_stacked_[j][labels_stacked_[j]>1].shape[0]>0)):

                            labels_stacked=labels_stacked_
                            #breakpoint()
                            #print('masksi: ', masksi)
                            labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                            masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                            boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                            
                        else:
                            #breakpoint()
                            numbeOfPixels=masks_mixed[j][masksi][masks_mixed[j][masksi]>0].shape[0]
                            numberOfOverlaps=masks_mixed[j][masksi][labels_stacked_[j]>1].shape[0]
                            percentage=numberOfOverlaps/numbeOfPixels*100

                            masks_mixed[j][masksi][labels_stacked_[j]>1]=0
                            #breakpoint()
                            if(torch.unique(masks_mixed[j][masksi]).shape[0]>1):
                                labels_stacked_[j][labels_stacked_[j]>1]=1
                                labels_stacked=labels_stacked_
                                

                                minIndx=np.min(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 0])
                                maxIndx=np.max(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 0])
                                minIndy=np.min(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 1])
                                maxIndy=np.max(np.argwhere(masks_mixed[j][masksi].cpu().numpy()>0)[:, 1])

                                labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                                masks_mixed_corrected[j].append(masks_mixed[j][masksi])
                                boxes_mixed_corrected[j].append(torch.tensor([minIndy,minIndx, maxIndy, maxIndx ]).to('cuda'))

                    
                        
                #breakpoint()  
                masks_mixed_corrected[j]=torch.stack(masks_mixed_corrected[j]).cpu().numpy() 
                masks_mixed_corrected[j]=BitmapMasks(masks_mixed_corrected[j],masks_mixed_corrected[j].shape[1], masks_mixed_corrected[j].shape[2])
                boxes_mixed_corrected[j]=torch.stack(boxes_mixed_corrected[j])
                labels_mixed_corrected[j]=torch.stack(labels_mixed_corrected[j])
        ##################################################################################################################
        
        labels_stacked=[]
        q_true=[0,0]
        labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))
        labels_stacked.append(torch.zeros(img.shape[2], img.shape[3]))
        if(not passThis):   
            for j in range(2):

                #print('masks_mixed len: ', len(masks_mixed))
                for masksi in range(all_masks_forVis_[j].shape[0]-1,-1,-1):
                    #breakpoint()
                    #labels_stacked_0[masksi]
                    #breakpoint()
                    #print('masksi: ', masksi)
                    if(masksi >= gt_bboxes[j].shape[0]):
                        labels_stacked[j][all_masks_forVis_[j][masksi]==1]+=1
                        labels_stacked[j][labels_stacked[j]>1]=1
                        #labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                        masks_forVis_corrected[j].append(all_masks_forVis_[j][masksi])
                        conf_forVis_corrected[j].append(all_conf_forVis_[j][masksi])
                        #boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                    else:
                        labels_stacked_=copy.deepcopy(labels_stacked)
                        labels_stacked_[j][all_masks_forVis_[j][masksi]==1]+=1
                        #breakpoint()
                        if(not(labels_stacked_[j][labels_stacked_[j]>1].shape[0]>0)):

                            labels_stacked=labels_stacked_
                            #breakpoint()
                            #print('masksi: ', masksi)
                            #labels_mixed_corrected[j].append(labels_mixed[j][masksi])
                            masks_forVis_corrected[j].append(all_masks_forVis_[j][masksi])
                            conf_forVis_corrected[j].append(all_conf_forVis_[j][masksi])
                            q_true[j]+=1
                            #boxes_mixed_corrected[j].append(boxes_mixed[j][masksi])
                        else:
                            #breakpoint()
                            numbeOfPixels=all_masks_forVis_[j][masksi][all_masks_forVis_[j][masksi]>0].shape[0]
                            numberOfOverlaps=all_masks_forVis_[j][masksi][labels_stacked_[j]>1].shape[0]
                            percentage=numberOfOverlaps/numbeOfPixels*100

                            all_masks_forVis_[j][masksi][labels_stacked_[j]>1]=0
                            #breakpoint()
                            if(torch.unique(all_masks_forVis_[j][masksi]).shape[0]>1):
                                labels_stacked_[j][labels_stacked_[j]>1]=1
                                labels_stacked=labels_stacked_
                                
                                masks_forVis_corrected[j].append(all_masks_forVis_[j][masksi])
                                conf_forVis_corrected[j].append(all_conf_forVis_[j][masksi])
                                q_true[j]+=1
                                #boxes_mixed_corrected[j].append(torch.tensor([minIndy,minIndx, maxIndy, maxIndx ]).to('cuda'))

                    
                        
                #breakpoint()  
                masks_forVis_corrected[j]=torch.stack(masks_forVis_corrected[j])#.cpu().numpy() 
                conf_forVis_corrected[j]=torch.stack(conf_forVis_corrected[j])#.cpu().numpy() 
                #masks_mixed_corrected[j]=BitmapMasks(masks_mixed_corrected[j],masks_mixed_corrected[j].shape[1], masks_mixed_corrected[j].shape[2])
                #boxes_mixed_corrected[j]=torch.stack(boxes_mixed_corrected[j])
                #labels_mixed_corrected[j]=torch.stack(labels_mixed_corrected[j])
                
                
            #breakpoint()
            #print('delete: ', delete) 
            #breakpoint()

        #gt_masks_inst[0]=BitmapMasks(all_masks0.cpu().numpy(), rolled_masks.shape[1], rolled_masks.shape[2])
                
                        #breakpoint()
                        #print('here')
                        
                        #breakpoint()
        visual=True
        #breakpoint()
        IMG_MEAN = [123.675/255., 116.28/255., 103.53/255.]
        IMG_STD = [58.395/255., 57.12/255., 57.375/255.]
        def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
                # 3, H, W, B
                #breakpoint()
                x=x.unsqueeze(3)
                ten = x.clone()#.permute(1, 2, 3, 0)
                for t, m, s in zip(ten, mean, std):
                    t.mul_(s).add_(m)
                # B, 3, H, W
                #breakpoint()
                return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2).squeeze(0)
        img_pastes=[]
        img_masksInstTarget=[]
        img_masksSemSource=[]
        img_insts_thres1=[]
        img_insts_thres2=[]
        img_insts_thres1_target=[]
        img_insts_thres2_target=[]
        if(visual and not passThis):
            #breakpoint()
            img_pastes.append(copy.deepcopy(mix_instance_img[0]))
            img_pastes.append(copy.deepcopy(mix_instance_img[1]))

            
            img_insts_thres1.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres2.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres1.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres2.append(torch.zeros((512,512)).to('cuda'))

            
            img_insts_thres1_target.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres1_target.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres2_target.append(torch.zeros((512,512)).to('cuda'))
            img_insts_thres2_target.append(torch.zeros((512,512)).to('cuda'))


            img_masksInstTarget.append(torch.zeros(mix_instance_img[0].shape).to('cuda'))
            img_masksInstTarget.append(torch.zeros(mix_instance_img[1].shape).to('cuda'))

            img_masksSemSource.append(torch.zeros(mix_instance_img[1].shape).to('cuda'))
            img_masksSemSource.append(torch.zeros(mix_instance_img[1].shape).to('cuda'))
            #breakpoint()
            from torchvision.utils import save_image

            ################################ source0

            mean=torch.mean(img[0].reshape(3,512*512), axis=1)
            std=torch.std(img[0].reshape(3,512*512), axis=1)
            mean=[123.675/255., 116.28/255., 103.53/255.]
            std=[58.395/255., 57.12/255., 57.375/255.]
            trans = transforms.Compose([ transforms.Normalize(mean = [ mean[0], mean[1], mean[2] ],
                                                     std = [ std[0], std[1], std[2] ]) ])
            #breakpoint()
            
            
            
            #transform = Denormalize(mean=(123.675/255., 116.28/255., 103.53/255.), std=(58.395/255., 57.12/255., 57.375/255.))
            #breakpoint()
            
            save_image(denormalize(img[0])*2, repo+'source0'+'_'+str(self.iter)+'.png')
            #breakpoint()
            #save_image(torch.clamp((trans(img[0])/2.)+0.5, 0, 1), repo+'source0'+'_'+str(self.iter)+'.png')

            #################################  source1

            mean2=torch.mean(img[1].reshape(3,512*512), axis=1)
            std2=torch.std(img[1].reshape(3,512*512), axis=1)
            #mean2=[123.675/255., 116.28/255., 103.53/255.]
            #std2=[58.395/255., 57.12/255., 57.375/255.]
            trans2 = transforms.Compose([ transforms.Normalize(mean = [ mean2[0], mean2[1], mean2[2] ],
                                                     std = [ std2[0], std2[1], std2[2] ]) ])
            #breakpoint()
            #save_image(torch.clamp((trans2(img[1])/2.)+0.5, 0, 1), repo+'source1'+'_'+str(self.iter)+'.png')
            save_image(denormalize(img[1])*2, repo+'source1'+'_'+str(self.iter)+'.png')

            ################################# target 0

            mean=torch.mean(target_img[0].reshape(3,512*512), axis=1)
            std=torch.std(target_img[0].reshape(3,512*512), axis=1)
            trans = transforms.Compose([ transforms.Normalize(mean = [ mean[0], mean[1], mean[2] ],
                                                     std = [ std[0], std[1], std[2] ]) ])
            #breakpoint()
            #save_image(torch.clamp((trans(target_img[0])/2.)+0.5, 0, 1), repo+'target0'+'_'+str(self.iter)+'.png')
            save_image(denormalize(target_img[0])*2, repo+'target0'+'_'+str(self.iter)+'.png')
            ################################## target 1
            mean1=torch.mean(target_img[1].reshape(3,512*512), axis=1)
            std1=torch.std(target_img[1].reshape(3,512*512), axis=1)
            trans1 = transforms.Compose([ transforms.Normalize(mean = [ mean1[0], mean1[1], mean1[2] ],
                                                     std = [ std1[0], std1[1], std1[2] ]) ])
            #breakpoint()
            #save_image(torch.clamp((trans1(target_img[1])/2.)+0.5, 0, 1), repo+'target1'+'_'+str(self.iter)+'.png')
            save_image(denormalize(target_img[1])*2, repo+'target1'+'_'+str(self.iter)+'.png')
            ################################## mixed with Semantic
            #breakpoint()
            img_mix_half1= mixed_img2[1]*mix_masks2[1].squeeze(0)
            img_mix_half2= mixed_img2[1]*(1-mix_masks2[1]).squeeze(0)


            #final_mixed_sem=  torch.clamp((trans1(target_img[1])/2.)+0.5, 0, 1)*mix_masks2[1].squeeze(0) + torch.clamp((trans2(img[1])/2.)+0.5, 0, 1)*(1-mix_masks2[1]).squeeze(0)  #+
            final_mixed_sem=  denormalize(target_img[1])*2*mix_masks2[1].squeeze(0) + denormalize(img[1])*2*(1-mix_masks2[1]).squeeze(0)  #+
            #breakpoint()
            save_image(final_mixed_sem ,repo+'mixed_semantic'+'_'+str(self.iter)+'.png')

            


            for j in range(batch_size):  
                save_image((img_pastes[j]+2.)/2., repo+'mixed_'+str(j)+'_'+str(self.iter)+'.png')
            #breakpoint()
            
            j=1
            for q in range(boxes_mixed_corrected[j].shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br = boxes_mixed_corrected[j][q]#proposal_list[0][i]
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
            img_pastes_maskOrig=copy.deepcopy(img)
            for q in range(gt_masks[j].masks.shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    #y_tl, x_tl, y_br, x_br = gt_masks[j].masks[q]#proposal_list[0][i]
                    #breakpoint()
                    minIndx=np.min(np.argwhere(gt_masks[j].masks[q]>0)[:, 0])
                    maxIndx=np.max(np.argwhere(gt_masks[j].masks[q]>0)[:, 0])

                    minIndy=np.min(np.argwhere(gt_masks[j].masks[q]>0)[:, 1])
                    maxIndy=np.max(np.argwhere(gt_masks[j].masks[q]>0)[:, 1])

                    #print('######### y: ',abs(y_tl-y_br), 'x: ', abs(x_tl-x_br))
                    #if(abs(y_tl-y_br) > 85 and abs(x_tl-x_br) > 85):
                    x_tl = minIndx
                    y_tl = minIndy
                    x_br = maxIndx
                    y_br = maxIndy
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    #img_pastes_maskOrig[j][:,x_tl:x_tl + 2, y_tl:y_br] = col
                    #img_pastes_maskOrig[j][:,x_br:x_br + 2, y_tl:y_br] = col
                    #img_pastes_maskOrig[j][:,x_tl:x_br, y_tl:y_tl + 2] = col
                    #img_pastes_maskOrig[j][:,x_tl:x_br, y_br:y_br + 2] = col
            #save_image((img_pastes_maskOrig[j]+2.)/2., repo+'carsMasks_orig_'+str(self.iter)+'.png')

            for q in range(gt_bboxes[j].shape[0]):
                
                if(True):
                    #breakpoint()
                    col=0
                    #breakpoint()
                    y_tl, x_tl, y_br, x_br = gt_bboxes[j][q]#proposal_list[0][i]
                    #print('######### y: ',abs(y_tl-y_br), 'x: ', abs(x_tl-x_br))
                    #if(abs(y_tl-y_br) > 85 and abs(x_tl-x_br) > 85):
                    x_tl = int(x_tl)
                    y_tl = int(y_tl)
                    x_br = int(x_br)
                    y_br = int(y_br)
                    #print('y_tl ; ', y_tl, 'x_tl: ', x_tl, 'x_br: ', x_br, ' y_br: ', y_br)
                    #print('imssssss: ', img)
                    #img[j][:,x_tl:x_tl + 2, y_tl:y_br] = col
                    #img[j][:,x_br:x_br + 2, y_tl:y_br] = col
                    #img[j][:,x_tl:x_br, y_tl:y_tl + 2] = col
                    #img[j][:,x_tl:x_br, y_br:y_br + 2] = col
            #save_image((img[j]+2.)/2., repo+'carsBoxes_orig_'+str(self.iter)+'.png')
            #breakpoint()
            #breakpoint()
            masks=torch.tensor(masks_mixed_corrected[j].masks).to('cuda')
            #breakpoint()

            
            #inds=gt_labels_inst[]
            for q in range(masks.shape[0]):
                #breakpoint()
                #if(gt_labels_inst[1][q]==2):
                img_pastes[j]=(1-masks[q].unsqueeze(0).repeat(3,1,1))*img_pastes[j]
            save_image((img_pastes[j]+2)/2., repo+'carsMasks_'+str(self.iter)+'.png')

            id_inst_other=1
            id_inst=1
            mix_instance_img_thre1=[]
            mix_instance_img_thre1.append(copy.deepcopy(img[0]))
            mix_instance_img_thre1.append(copy.deepcopy(img[1]))

            mix_instance_img_thre2=[]
            mix_instance_img_thre2.append(copy.deepcopy(denormalize(img[0])*255.))
            mix_instance_img_thre2.append(copy.deepcopy(denormalize(img[1])*255.))
            oneTime=False
 
            #breakpoint()
            for q in range(masks_forVis_corrected[j].shape[0]):
                #breakpoint()
                #if(gt_labels_inst[1][q]==2):
                #img_insts[j]+=masks[q].to('cuda')*id_inst
                #breakpoint()
                if(conf_forVis_corrected[j][q] > thres2):#gt
                    if q>=(masks_forVis_corrected[j].shape[0]-q_true[j]):
                        img_insts_thres2[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst_other
                        id_inst_other+=1
                    #breakpoint()
                    if q<(masks_forVis_corrected[j].shape[0]-q_true[j]):
                        #if(oneTime==False):
                        if(self.iter==1 and q==10):
                            print('none')
                            oneTime=True
                        else:
                            img_insts_thres2[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst_other
                            mix_instance_img_thre2[j]=masks_forVis_corrected[j][q].repeat(3,1,1)*denormalize(target_img[j])*255.+(1-masks_forVis_corrected[j][q].repeat(3,1,1))*mix_instance_img_thre2[j]
                            img_insts_thres2_target[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst_other
                            id_inst_other+=1
                    #img_insts[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
            
                #if(conf_forVis_corrected[j][q] > thres2):
                #
                #
                #    if q<(masks_forVis_corrected[j].shape[0]-q_true[j]):
                #        #breakpoint()
                #        if(oneTime==False):
                #            print('none')
                #            oneTime=True
                #        else:
                #            img_insts_thres2[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                #            mix_instance_img_thre2[j]=masks_forVis_corrected[j][q].repeat(3,1,1)*target_img[j]+(1-masks_forVis_corrected[j][q].repeat(3,1,1))*mix_instance_img_thre2[j][j]
                #            img_insts_thres2_target[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                #    else:
                #        img_insts_thres2[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                    
                if(conf_forVis_corrected[j][q] > thres1):#0.5
                    img_insts_thres1[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                    #breakpoint()
                    if q<(masks_forVis_corrected[j].shape[0]-q_true[j]):
                        #breakpoint()
                        mix_instance_img_thre1[j]=masks_forVis_corrected[j][q].repeat(3,1,1)*target_img[j]+(1-masks_forVis_corrected[j][q].repeat(3,1,1))*mix_instance_img_thre1[j]
                        img_insts_thres1_target[j]+=masks_forVis_corrected[j][q].to('cuda')*id_inst
                    id_inst+=1
            #breakpoint()
            
            #breakpoint()
            img_insts_thres2[j]=img_insts_thres2[j].int()#unsqueeze(0).
            img_insts_thres1[j]=img_insts_thres1[j].int()
            img_insts_thres1_target[j]=img_insts_thres1_target[j].int()
            img_insts_thres2_target[j]=img_insts_thres2_target[j].int()
            id=0.1

            #all_masks_ForVis_.append(torch.cat([torch.tensor(gt_masks[j].masks).to('cuda'), torch.stack(all_masks_ForVis).to('cuda')]))
            #all_conf_forVis
            #breakpoint()
            #mmmmmmm
            self.runOtherDirection(masks_forVis_corrected,conf_forVis_corrected,q_true, thres1, thres2, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas, mixed_img2, mix_masks2)
            #if(q > gt_bboxes[j].shape[0] and oneTime==False):
            #oneTime=False
            for q in range(masks_forVis_corrected[j].shape[0]):
                #breakpoint()
                #if(gt_labels_inst[1][q]==2):
                #img_pastes[j]=(1-masks[q].unsqueeze(0).repeat(3,1,1))*img_pastes[j]
                if(conf_forVis_corrected[j][q] > thres2):

                    img_pastes[j]=(1-masks_forVis_corrected[1][q].unsqueeze(0).repeat(3,1,1))*img_pastes[j]
                    #breakpoint()
                    #img_masksInstTarget[j]=(masks[q].unsqueeze(0).repeat(3,1,1))*id+img_masksInstTarget[j]
                    img_masksInstTarget[j]=(masks_forVis_corrected[1][q].unsqueeze(0).repeat(3,1,1))*id+img_masksInstTarget[j]
                    id+=0.1
                #breakpoint()
            

            #denormalize = transforms.Normalize((-1 * IMG_MEAN / IMG_VAR), (1.0 / IMG_VAR))
            #breakpoint()
            mean=torch.mean(mix_instance_img_thre2[j].reshape(3,512*512), axis=1)
            std=torch.std(mix_instance_img_thre2[j].reshape(3,512*512), axis=1)
            trans = transforms.Compose([ transforms.Normalize(mean = [ mean[0], mean[1], mean[2] ],
                                                     std = [ std[0], std[1], std[2] ]) ])
            #mix_instance_img_thre2[j] = denormalize(mix_instance_img_thre2[j])*255.
            #breakpoint()
            mix_instance_img_thre2[j]*=2
            #mix_instance_img_thre2[j]=torch.clamp((mix_instance_img_thre2[j]/2.)+0.5, 0, 1)*255.
            #mix_instance_img_thre2[j] = denormalize(mix_instance_img_thre2[j])*255.
            mean=torch.mean(mix_instance_img_thre1[j].reshape(3,512*512), axis=1)
            std=torch.std(mix_instance_img_thre1[j].reshape(3,512*512), axis=1)
            trans = transforms.Compose([ transforms.Normalize(mean = [ mean[0], mean[1], mean[2] ],
                                                     std = [ std[0], std[1], std[2] ]) ])
            #mix_instance_img_thre1[j] = trans(mix_instance_img_thre1[j])
            mix_instance_img_thre1[j] = denormalize(mix_instance_img_thre1[j])*255.#breakpoint()
            #mix_instance_img_thre1[j]=torch.clamp((mix_instance_img_thre1[j]/2.)+0.5, 0, 1)*255.
            #breakpoint()
            
            mean=torch.mean(target_img[j].reshape(3,512*512), axis=1)
            std=torch.std(target_img[j].reshape(3,512*512), axis=1)
            transTarget = transforms.Compose([ transforms.Normalize(mean = [ mean[0], mean[1], mean[2] ],
                                                     std = [ std[0], std[1], std[2] ]) ])
            #breakpoint()
            
            #mix_instance_img[j] = invTrans(mix_instance_img[j].permute(1,2,0))
            #mix_instance_img[j]=mix_instance_img[j].permute(2,0,1)
            #breakpoint()
            img_new1, img_new2, img_new_target1, img_new_target2=prep_ins_for_vis_mine(img_insts_thres1[j].cpu(), img_insts_thres2[j].cpu(), img_insts_thres1_target[j].cpu(), img_insts_thres2_target[j].cpu(), mode='GT', dataset_name='synthia', debug=False, blend_ratio=0.5, img1=mix_instance_img_thre1[j].cpu(),img2=mix_instance_img_thre2[j].cpu(), target=(torch.clamp((transTarget(target_img[j])/2.)+0.5, 0, 1)*255.).cpu(), show_only_person=False, sem_id=None)
            #mix_instance_img[j]=img_new/255.
            #breakpoint()

            save_image((img_pastes[j]+2)/2., repo+'carsMasks_'+str(self.iter)+'.png')
            save_image((img_masksInstTarget[j]), repo+'maskInstTargets_'+str(j)+'_'+str(self.iter)+'.png')
            save_image(img_new1/255., repo+'mixedBefore_'+str(j)+'_'+str(self.iter)+'.png')
            save_image(img_new2/255., repo+'mixedAfter_'+str(j)+'_'+str(self.iter)+'.png')
            save_image(img_new_target1/255., repo+'targetBefore_'+str(j)+'_'+str(self.iter)+'.png')
            save_image(img_new_target2/255., repo+'targetAfter_'+str(j)+'_'+str(self.iter)+'.png')
            
        #breakpoint()
        #print('here')
        #if(True in list(maskSizes0[gt_labels[0]==2]>50000)):
        #if(not passThis):
            #print('boxes mixed: ', boxes_mixed_corrected[j].shape)
            #print('boxes orig: ', gt_bboxes[j].shape)
            #print('mix_instance_img[0] shape: ', mix_instance_img[0].shape)
            #print('mix_instance_img[1] shape: ', mix_instance_img[1].shape)
            #print('img shape: ', img.shape)
            #breakpoint()
            #inst_losses = self.get_model().forward_train(torch.stack(mix_instance_img),
            #                                    img_metas,
            #                                    gt_semantic_seg=None,
            #                                    return_feat=True,
            #                                    gt_bboxes=boxes_mixed_corrected,
            #                                    gt_labels=labels_mixed_corrected,
            #                                    gt_masks=masks_mixed_corrected,
            #                                    )

            #inst_losses.pop('features')
            #inst_losses = add_prefix(inst_losses, 'inst_mixed')
            #inst_loss, inst_log_vars = self._parse_losses(inst_losses)
            #log_vars.update(inst_log_vars)
            #inst_loss.backward()
            #print('after')
   
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
        #breakpoint()
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
        #breakpoint()

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
        mixed_img2=[None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        ps_target_seg=pseudo_label.unsqueeze(1)
        mix_masks2 = get_class_masks(ps_target_seg)
        #breakpoint()

        if self.disable_mix_masks:
            for i in range(batch_size):
                mix_masks[i][:] = 0
                assert mix_masks[i].sum() == 0, 'problem found'

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(strong_parameters, data=torch.stack((img[i], target_img[i])), target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks2[i]
            mixed_img2[i], _ = strong_transform(strong_parameters, data=torch.stack((target_img[i], img[i])), target=torch.stack((pseudo_label[i], gt_semantic_seg[i][0])))

            
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mixed_img2 = torch.cat(mixed_img2)

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


        #self.runOtherDirection(0., 0.95, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas, mixed_img2, mix_masks2)
        self.runInstances(0., 0.95, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas, mixed_img2, mix_masks2)
        #self.runInstances(0.95, img, target_img, target_img_metas, gt_bboxes, gt_masks, gt_labels, batch_size, img_metas)
        self.iter+=1

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

        return log_vars

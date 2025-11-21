#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:17:37 2024

@author: yjadvance

Gradient flow tracking from Li et al.
"""

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import skimage
import skimage.morphology as mp
from skimage import measure as ms
from torchvision.ops import masks_to_boxes
import copy

from torchvision.transforms import ToTensor, GaussianBlur
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from skimage.feature import blob_dog, blob_log, blob_doh
import scipy
from skimage.segmentation import watershed


class PatchEmbedding(nn.Module):
    def __init__(self,image_size,patch_size,hidden_size):
        super(PatchEmbedding,self).__init__()
        if len(image_size)==4:
            n,c,h,w = image_size
        elif len(image_size)==5:
            n,c,h,w= image_size[0],image_size[2],image_size[3],image_size[4]
        # self.patches= torch.zeros(n,patch_size**2,h*w*c//patch_size**2)
        self.n_patches = h // patch_size
        self.projection = nn.Conv2d(c,hidden_size,kernel_size=self.n_patches,stride=self.n_patches)
        
    def forward(self,x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1,2) #5,3025,201
        return x
    

def get_positional_embeddings(sequence_length,d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/10000**(j/d)) if j%2 ==0 else np.cos(i/10000**((j-1)/d))
    return result

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d,n_heads=2):
        super(MultiHeadSelfAttention,self).__init__()
        self.d = d
        self.n_heads = n_heads
        # print('d: ',d,' n_head: ',n_heads)
        assert d % n_heads ==0
        
        d_head = int(d/n_heads)
        
        self.q_mappings = nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, sequences,guided_attention=None):
        '''
        Sequences has shape (N, seq_length, token_dim)
        We go into shape    (N, seq_length, n_heads, token_dim/n_heads)
        And come back to    (N, seq_length, item_dim) through concatenation
        '''
        
        result = []
        attention_all = []
        for idx,sequence in enumerate(sequences):
            seq_result = []
            attention_cur=[]
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                
                seq = sequence[:,head*self.d_head:(head+1)*self.d_head]
                q,k,v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                

                if guided_attention==None:
                    attention = self.softmax(q@k.T / (self.d_head**0.5))
                else:
                    attention = self.softmax(q@k.T / (self.d_head**0.5))
                    attention_temp=attention.clone()
                    guided_attention_cur=guided_attention[idx]
                    guided_attention_cur=guided_attention_cur.flatten().unsqueeze(1)
                    guided_attention_cur=guided_attention_cur.repeat(1,attention.size(1))##For some reason, repeat (0,size(1)) doesn't work as I expected
                    guided_attention_cur=guided_attention_cur.permute((1,0))
                    attention_vision_part_normalized=torch.sum(attention[:,attention.size(0)-guided_attention_cur.size(1):],dim=1)
                    # guided_attention_cur=guided_attention_cur*torch.div(attention_vision_part_normalized,torch.sum(guided_attention_cur,dim=1))
                    attention_vision_part_normalized=torch.div(attention_vision_part_normalized,torch.sum(guided_attention_cur,dim=1)).reshape(-1,1).repeat(1,guided_attention_cur.size(1))
                    guided_attention_cur=guided_attention_cur*attention_vision_part_normalized
                    
                    attention_temp[:,attention.size(0)-guided_attention_cur.size(1):]=guided_attention_cur
                    attention=attention_temp
                    # print(attention.device)
                attention_cur.append(attention)
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
            # attention_all.append(torch.mean(torch.stack(attention_cur),dim=0))
            attention_all.append(torch.stack(attention_cur))
        # att=torch.stack(attention_all,dim=0)
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result]), torch.cat([torch.unsqueeze(a, dim=0) for a in attention_all])
            
class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio = 4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadSelfAttention(hidden_d,n_heads)
        
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d, hidden_d)
            )
    
    def forward(self, x,guided_attention=None):
        if guided_attention==None:
            out, attention =self.mhsa(self.norm1(x))
        else:
            out, attention =self.mhsa(self.norm1(x),guided_attention)
        out = x+ out
        out = out+self.mlp(self.norm2(out))
        return out, attention
    


class VGT(nn.Module):
    def __init__(self,nchw=(1,1,28,28),patch_size=7,n_batches=10, n_blocks = 2, hidden_d = 8, n_heads =2,n_genes=117, out_d=10,device='cpu',threshold=0,return_attention=True,debug_info=False):
        super(VGT,self).__init__()
        self.nchw =nchw
        self.n_patches = (nchw[2]/patch_size, nchw[3]/patch_size)
        # self.n_batches = n_batches
        self.hidden_d = hidden_d
        self.patch_size=patch_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_genes = n_genes
        self.device=device
        self.return_attention = return_attention
        self.debug_info = debug_info
        self.input_d = int(nchw[1]*self.n_patches[0] * self.n_patches[1])
        self.patchify = PatchEmbedding(self.nchw,patch_size,hidden_d)

        self.threshold = threshold
        ###Learnable class token    
        self.class_token = nn.Parameter(torch.rand(self.n_genes,self.hidden_d))
        ###Positional embedding
        # self.pos_embed = nn.Parameter(torch.rand(1,self.patch_size**2+self.n_genes, self.hidden_d))]
        self.pos_embed = nn.Parameter(torch.rand(1,15+self.n_genes, self.hidden_d))
        #self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.patch_size**2+self.n_genes, self.hidden_d)))
        ###Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        

        self.gene_expression_many_to_one=nn.Sequential(
            nn.Linear(self.hidden_d,1),

            )        
        self.summation = nn.Linear(15,1)
        self.gene_expression_many_to_one1=nn.Sequential(

            nn.Linear(self.hidden_d,self.n_genes),
            nn.ReLU(),            

            )
        
        self.RegressionSolver=nn.Bilinear(in1_features=self.hidden_d,in2_features=self.hidden_d,out_features=self.n_genes)
        
        
        self.avg_pool = nn.AvgPool2d((3,3), stride=1,padding=1)

    def RoundFloat(self,x):
        if x>=0:
            t = torch.ceil(x)
            if t-x>0.5:
                t -=1.0
            return t
        else:
            t = torch.ceil(-x)
            if t+x > 0.5:
                t -=1.0
            return -t
    
    def DetectCycle(self,trajectory, points,z):
        ##Initialize trajectory length
        length = 0
    
        ##Identify trajectory bounding box
        # xMin = torch.min(trajectory[int(z),0:points,0])
        # xMax = torch.max(trajectory[int(z),0:points,0])
        xMin = torch.min(torch.from_numpy(trajectory[int(z),0:points,0]))
        xMax = torch.max(torch.from_numpy(trajectory[int(z),0:points,0]))

        xRange = xMax - xMin +1
    
        # yMin = torch.min(trajectory[int(z),0:points,1])
        # yMax = torch.max(trajectory[int(z),0:points,1])
        yMin = torch.min(torch.from_numpy(trajectory[int(z),0:points,1]))
        yMax = torch.max(torch.from_numpy(trajectory[int(z),0:points,1]))

        yRange = yMax - yMin +1
        
        ##Fill in the trajectory map
        trajectory_map = torch.zeros(int(xRange),int(yRange))
        for i in range(points):
            if trajectory_map[int(trajectory[int(z),i,0]-xMin), int(trajectory[int(z),i,1]-yMin)] ==1:
                break
            else:
                trajectory_map[int(trajectory[int(z),i,0]-xMin), int(trajectory[int(z),i,1]-yMin)] = 1
            length +=1
        
        return length
        

    def first_nonzero_pos(self,x):
        s = f"{x:.16f}".split('.')[1]  # decimal part as string
        for i, ch in enumerate(s, 1):
            if ch != '0':
                return i
        return 0  # in case all digits are zero
    
    def floats_below_digit(self,nums):
        # find first nonzero digit position per float
        positions = [self.first_nonzero_pos(x) for x in nums]
        min_pos = min(p for p in positions if p > 0)
        below_idx = [i for i, p in enumerate(positions) if p > min_pos]
        return min_pos, below_idx
        
        
        
    
    
    def AttentionGradientFlowTracking(self,attention, bin_size,threshold, K=50,epoch_cur=None,batch_idx=None,images=None):
        '''
        Pytorch implementation of G. Li et al "3D cell nuclei segmentation based on gradient flow tracking" in BMC Cell Biology, vol40, no.8, 2007
    
        attention: (AH,N,W,H) where N = num batches, AH = Num heads, M = num genes, W = Width, 
        '''
        attention = attention.detach().cpu()
        factor=1
        num_batch=attention.size(1)
        ##proposals=attention_all[:,:,:,:self.n_genes,self.n_genes:]
        attention=attention[:,:,:,:,self.n_genes:]
        attention=attention.reshape(attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3],self.patch_size,self.patch_size)
        width,height = attention.shape[4],attention.shape[5]

        attention_merged=torch.mean(attention,dim=[0,2,3])
        attention_merged=self.avg_pool(attention_merged)
        grad_x,grad_y=torch.gradient(attention_merged,dim=[1,2])


        
        magnitude = torch.sqrt((grad_x**2+grad_y**2)+torch.finfo(float).eps)        
        grad_x = grad_x/magnitude
        grad_y = grad_y/magnitude
                
        
        ##Define mask to track pixels that are mapped to a sink
        mapped = torch.zeros(attention_merged.shape)        ##Define label image
        segmentation = torch.zeros(attention_merged.shape)
        segmentation_dict = [dict([]) for _ in range(num_batch)]
        ##Initialize a list of sinks
        sinks=[[] for _ in range(num_batch)]
        
        ##Define coordinates for foreground pixels(mask ==1)
        ######TODO: Order gradient strength. Ascending order since sink will have less gradient changes
        
        n,i,j=[],[],[]
        
        for idx in range(len(magnitude)):##Later change it to num batches
            x_cur,y_cur = torch.meshgrid(torch.arange(magnitude[idx].size(0)),torch.arange(magnitude[idx].size(1)))
            attention_sum_flatten=magnitude[idx].flatten()
            x_cur=x_cur.flatten().to('cuda:0')
            y_cur=y_cur.flatten().to('cuda:0')
            attention_sum_order_index=torch.argsort(attention_sum_flatten,descending=False)
            x_cur=x_cur[attention_sum_order_index]*factor###!!! We don't need to look for all points in the stretched one
            y_cur=y_cur[attention_sum_order_index]*factor
            n+=idx*torch.ones(len(x_cur)).to('cuda:0')
            i+=x_cur
            j+=y_cur          
        
        last_batch=-1
        # z=torch.tensor(0).to('cuda')
        # x=2
        # y=18
        cosphi_map = np.zeros(attention_merged.shape)
        cosphi_map_novel = np.zeros(attention_merged.shape)
        cosphi_map_both_neg = np.zeros(attention_merged.shape)
        strength_map_x = np.zeros(attention_merged.shape)
        strength_map_y = np.zeros(attention_merged.shape)
        point_1_map=np.zeros(attention_merged.shape)
        for index, (z,x,y) in enumerate(zip(n,i,j)):
        # for index in range(1):
            
            ##Initialize angle, trajectory length, novel flag, and allocation count
            cosphi = 1
            points = 1
            novel = 1
            ##alloc should be updated per batch
            if last_batch !=z:
                alloc = 1
            last_batch=z
            is_increasing=1
            
            ##Initialize trajectory
            # trajectory = torch.zeros((num_batch,K,2))#.to('cuda:0')
            trajectory = np.zeros((num_batch,K,2))            
            trajectory[int(z),0,0] = int(x)
            trajectory[int(z),0,1] = int(y)
            last_xstep = 0
            last_ystep = 0
            
            ###If we traveled all mapped space, we should stop the algo for the z-th batch.
            ###Current evaluation: sum(mapped)==mapped_height*mapped_width
            if torch.sum(mapped[int(z)])==mapped[int(z)].size(0)*mapped[int(z)].size(1):
                print(333)
                # continue
            
            ##Track while angle defined by successive steps is < torch.pi/2
            # while (cosphi >0) and (is_increasing>0):
            while (cosphi >0):
                
                ##Calculate step
                xStep = self.RoundFloat(grad_x[int(z),int(trajectory[int(z),points-1,0]), int(trajectory[int(z),points-1,1])])*bin_size
                yStep = self.RoundFloat(grad_y[int(z),int(trajectory[int(z),points-1,0]), int(trajectory[int(z),points-1,1])])*bin_size

                strength_map_x[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])]=grad_x[int(z),int(trajectory[int(z),points-1,0]), int(trajectory[int(z),points-1,1])]
                strength_map_y[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])]=grad_y[int(z),int(trajectory[int(z),points-1,0]), int(trajectory[int(z),points-1,1])]
                
                if grad_y[int(z),int(trajectory[int(z),points-1,0]), int(trajectory[int(z),points-1,1])] <0:
                    pass
                    # print(111)
                if xStep ==0 and yStep ==0 and last_xstep==0 and last_ystep==0:
                    novel = -1
                    break
                elif xStep ==0 and yStep ==0:
                    xStep = last_xstep
                    yStep = last_ystep
                
                last_xstep = xStep
                last_ystep = yStep
                
                ##Check the image edge            
                if (trajectory[int(z),points-1,0]+xStep<0) or (trajectory[int(z),points-1,0]+xStep>width-1) or (trajectory[int(z),points-1,1]+yStep<0) or (trajectory[int(z),points-1,1]+yStep>height-1):
                    break
                
                ##Add a new point to trajectory list
                if points < K: ##Buffer is not overrun
                    trajectory[int(z),points,0] = trajectory[int(z),points-1, 0] + xStep
                    trajectory[int(z),points,1] = trajectory[int(z),points-1, 1] + yStep
                    # print('tra: ',trajectory[int(z),points,0],trajectory[int(z),points,1])
                else: ##Buffer overrun
                    
                    ##Check for cycle
                    cycle = self.DetectCycle(trajectory,points,int(z))
                    
                    if cycle == points: ##No cycle, simple overflow. Grow buffer
                        
                        ##Copy and reallocate
                        temp = trajectory #(5,50,2)
                        trajectory = np.zeros((len(temp),2*temp.shape[1],2)) #(5,100,2)
                        print('K: ',K)
                        print('alloc: ',alloc)
                        print('temp: ',np.shape(temp))
                        print('trajectory: ',np.shape(trajectory))
                        trajectory[:,:temp.shape[1], ] = temp #
                        alloc +=1
                        
                        ##Add a new point
                        trajectory[int(z),points, 0] = trajectory[int(z),points-1, 0] +xStep
                        trajectory[int(z),points, 1] = trajectory[int(z),points-1, 1] +yStep
                    else:
                        points = cycle
                        # cosphi_map_novel[int(z),int(trajectory[int(z),points-1,0]),int(trajectory[int(z),points-1, 1])] = 10

                        break
            ##Check mapping
                if mapped[int(z), int(trajectory[int(z),points, 0]), int(trajectory[int(z),points, 1])] >0:
                    novel = 0
                    cosphi = -1
                    cosphi_map[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])]=cosphi
                    # cosphi_map_novel[int(z),int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])] = 10
                # elif mask[int(z), int(trajectory[int(z),points, 0]), int(trajectory[int(z),points, 1])] ==0:
                #     cosphi = -1
                else:
                    cosphi = grad_y[int(z),int(trajectory[int(z),points-1, 0]),int(trajectory[int(z),points-1, 1])]*grad_y[int(z),int(trajectory[int(z),points, 0]),int(trajectory[int(z),points, 1])] +grad_x[int(z),int(trajectory[int(z),points-1, 0]),int(trajectory[int(z),points-1, 1])]*grad_x[int(z),int(trajectory[int(z),points, 0]),int(trajectory[int(z),points, 1])]
                    cosphi_map[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])]=cosphi

                    

                    
                ##Increment trajectory length counter
                points +=1

                ##Added constraint - Stop if it's not hiking up
                # is_increasing = magnitude[int(z),int(trajectory[int(z),points, 0]),int(trajectory[int(z),points, 1])]-magnitude[int(z),int(trajectory[int(z),points-1, 0]),int(trajectory[int(z),points-1, 1])]
            
            ##Determine if sink is novel
            if novel ==1:
                
                ##Record sink
                # sinks.append(torch.cat((z,trajectory[points-1,])))
                # sinks.append(torch.cat((z.reshape(-1,1),trajectory[int(z),points-1,].reshape(2,1)))) ###!!!
                sinks[int(z)].append(trajectory[int(z),points-1,]) ###!!!
                mapped[int(z), int(trajectory[int(z),points-1,0]),int(trajectory[int(z),points-1,1])] = len(sinks[int(z)])
                ##Add trajectory to label image with new sink value, add mapping
                path_as_list=[]
                for j in range(points):
                    segmentation[int(z),int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])] = len(sinks[int(z)])
                    path_as_list.append([int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])])

                    # mapped[int(z), int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])] = len(sinks)
                segmentation_dict[int(z)][len(sinks[int(z)])]=[[path_as_list]]
                
                if points==1:
                    point_1_map[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])]=1
                
                # cosphi_map_novel[int(z),int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])] = 1
            elif novel ==0:
    
                ##Add trajectory to label image with sink value of final point
                # print(111)
                path_as_list=[]
                for j in range(points):
                    segmentation[int(z),int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])] = segmentation[int(z),int(trajectory[int(z),points-1,0]),int(trajectory[int(z),points-1,1])]
                    path_as_list.append([int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])])
                    # segmentation_dict[int(z)][int(mapped[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])])].append([int(trajectory[int(z),j,0]),int(trajectory[int(z),j,1])])
                segmentation_dict[int(z)][int(mapped[int(z), int(trajectory[int(z),points-1, 0]), int(trajectory[int(z),points-1, 1])])].append([path_as_list])

        
        #####Select 'strong' sinks
        def SelectStrongSinks(sinks,mapped,segmentation_dict,num_batch,threshold,guided_object_from_prev=None,dilated_from_prev=None,computed_cosphi_map=None):
        
            sinks_new=[[] for _ in range(num_batch)]
            mapped_new=torch.zeros(mapped.shape)
    
            ###Approach 2: we care the direction
            segmentation_dict_direction =  [dict([]) for _ in range(num_batch)]
            for i in range(num_batch):
                for j in range(len(sinks[i])):
                    key=int(mapped[i][int(sinks[i][j][0]),int(sinks[i][j][1])])
                    direction=np.zeros((3,3))
                    for k in range(len(segmentation_dict[i][key])):
                        ###If lenth ==1: we can consider it as a noise
                        if len(segmentation_dict[i][key][k][0])==1:
                            continue
                        ###Otherwise, always -1 is sink, so -2 is the point where it's coming to the sink
                        point_cur=np.asarray(segmentation_dict[i][key][k][0][-1])
                        direction_cur=np.asarray(segmentation_dict[i][key][k][0][-2])
                        direction_cur=direction_cur-point_cur
                        direction[1+direction_cur[0],1+direction_cur[1]]=1
                    segmentation_dict_direction[i][key]=np.sum(direction)
                    if np.sum(direction)>threshold:
                        sinks_new[i].append(sinks[i][j])
                        mapped_new[i][int(sinks[i][j][0]),int(sinks[i][j][1])]=key
                        
            ###Method with strong sink
            seedImage = np.zeros(segmentation.shape)
            erosion = np.zeros(segmentation.shape)
            dilated = np.zeros(segmentation.shape)
            
            guided_object=torch.zeros(segmentation.size())##Output
            labels = []
            for i in range(len(sinks_new)):##Batch level
                cur_neg_cosphi_map = computed_cosphi_map[i]<=0
                for j in range(len(sinks_new[i])):
                    seedImage[i][int(sinks_new[i][j][0]),int(sinks_new[i][j][1])] = j+1
        
                # We first sharpen our image with erosion
                # erosion[i] = mp.dilation(seedImage[i], mp.disk(1))
            
                # generate new labels for dilated seeds, define memberships using label func
                labeled,num_objects=scipy.ndimage.label(1*(seedImage[i]>0),structure=np.ones((3,3)))


                # region=regionprops(labeled)
                # hull=ConvexHull(region[0].coords)
                
                dilated[i] = mp.dilation(labeled,footprint=np.ones((3,3)))
                labels.append(num_objects)
                
                ##We remove outer rims
                outer_val = dilated[i][1,1]
                vals = np.unique(dilated[i])
                for val in vals:
                    cur_label_image = dilated[i] == val
                    cur_label_image  = np.nonzero(cur_label_image)
                    y_min,y_max = cur_label_image[0].min(),cur_label_image[0].max()
                    x_min,x_max = cur_label_image[1].min(),cur_label_image[1].max()
                    
                    # if (y_max-y_min)*(x_max-x_min)>65*65*0.8:
                    #     dilated[i][dilated[i] ==val] = 0
    
                ##Now for each sink and trajectory, we check if it's in the label. If yes, we mark them in the guided_obj
                test_map=np.zeros((len(sinks_new[0]),seedImage.shape[1],seedImage.shape[2]))
                for j in range(len(sinks_new[i])):
                    key=int(mapped[i][int(sinks_new[i][j][0]),int(sinks_new[i][j][1])])
                    for trajectory in segmentation_dict[i][key]:
                        # if len(trajectory[0])>5:
                        #     guided_object[i][guided_object[i]==dilated_from_prev[i][int(sinks_new[i][j][0]),int(sinks_new[i][j][1])]]=0
                        #     break
                        trajectory_map_single=np.zeros((seedImage.shape[1],seedImage.shape[2]))
                        for paths in trajectory:
                            for path in paths:
                                if isinstance(dilated_from_prev,np.ndarray):
                                    guided_object[i][path[0],path[1]]=dilated_from_prev[i][int(sinks_new[i][j][0]),int(sinks_new[i][j][1])]
                                    trajectory_map_single[path[0],path[1]]=1
                                    # test_map[j,path[0],path[1]]=1
                                else:
                                    guided_object[i][path[0],path[1]]=dilated[i][int(sinks_new[i][j][0]),int(sinks_new[i][j][1])]
                                    # test_map[j,path[0],path[1]]=1

            
            
            # test=cur_neg_cosphi_map+overlap
            
            return guided_object,seedImage,labels,dilated
        guided_object,seedImage,labels,dilated=SelectStrongSinks(sinks,mapped,segmentation_dict,num_batch,threshold=0,computed_cosphi_map=cosphi_map)
        guided_object1,seedImage1,labels1,dilated1=SelectStrongSinks(sinks,mapped,segmentation_dict,num_batch,threshold,guided_object,dilated_from_prev=dilated,computed_cosphi_map=cosphi_map)
        # guided_object,seedImage,labels,dilated=SelectStrongSinks(sinks,mapped,segmentation_dict,num_batch,threshold,guided_object,dilated_from_prev=dilated,computed_cosphi_map=cosphi_map)
        


        ####Generate images for qualitative analysis
        selected_image=np.asarray(images[0].detach().cpu().permute(1,2,0).numpy(),dtype=np.uint8)
        if self.debug_info:
            if batch_idx in [0,1,2,3,4,5,6,7,8,9,10]:
            # if batch_idx in [0,1,2,3,4,5,6,7,8,9,10]:
                ##attention_sum
                cur=attention_merged[0].detach()
                cur=torch.unsqueeze(cur, 0)
                cur=torch.unsqueeze(cur, 0)
                cur=F.interpolate(cur,size=(100,100),mode='bilinear',align_corners=False)
                cur=cur.squeeze(0)
                cur=cur.squeeze(0).clone().detach().cpu().numpy()
                plt.imsave(os.path.join(self.debug_info,'attention_sum','attention_sum_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'),cur)
                ###Selected sink
                cur=torch.from_numpy(seedImage[0]).detach()
                cur=torch.unsqueeze(cur, 0)
                cur=torch.unsqueeze(cur, 0)
                cur=F.interpolate(cur,size=(100,100),mode='nearest')
                cur=cur.squeeze(0)
                cur=cur.squeeze(0).clone().detach().cpu().numpy()
                plt.imsave(os.path.join(self.debug_info,'attention_sum','sink_strong_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'),cur)
                ##Guided object
                cur=guided_object[0].detach()
                cur=torch.unsqueeze(cur, 0)
                cur=torch.unsqueeze(cur, 0)
                cur=F.interpolate(cur,size=(100,100),mode='nearest')
                cur=cur.squeeze(0)
                cur=cur.squeeze(0).detach().cpu().numpy()
                plt.imsave(os.path.join(self.debug_info,'attention_sum','GuidedObject_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'),cur)
                ###All sinks with gradient flow
                # cur=mapped[0]
                # plt.imshow(cur)
                # x_grid,y_grid = np.meshgrid(np.arange(grad_x.size(1)),np.arange(grad_x.size(1)))
                # plt.quiver(x_grid,y_grid,grad_y[0].detach().cpu(),(-1)*grad_x[0].detach().cpu())
                # plt.savefig(os.path.join(self.debug_info,'attention_sum','sink_all_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'))
                # plt.clf()
                #Guided object with hist image as background
                cur=guided_object[0].detach()
                # cur=torch.unsqueeze(cur, 0)
                # cur=torch.unsqueeze(cur, 0)
                # cur=F.interpolate(cur,size=(65,65),mode='nearest')
                # cur=cur.squeeze(0)
                cur=cur.squeeze(0).detach().cpu().numpy()
                plt.imshow(selected_image)
                plt.imshow(cur,alpha=0.5)
                plt.savefig(os.path.join(self.debug_info,'attention_sum','Segmentation_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'))
                plt.clf()

                #Guided object with hist image as background
                cur=guided_object1[0].detach()
                # cur=torch.unsqueeze(cur, 0)
                # cur=torch.unsqueeze(cur, 0)
                # cur=F.interpolate(cur,size=(65,65),mode='nearest')
                # cur=cur.squeeze(0)
                cur=cur.squeeze(0).detach().cpu().numpy()
                plt.imshow(selected_image)
                plt.imshow(cur,alpha=0.5)
                plt.savefig(os.path.join(self.debug_info,'attention_sum','Segmentation1_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'))
                plt.clf()
                
        return guided_object1,labels,attention_merged, guided_object
    



    
    def forward(self, images,status,guided_attention=None,num_labels=None,epoch_cur=-1,batch_idx=None):
        if self.debug_info:
            if batch_idx in [0,1,2,3,4,5,6,7,8,9,10] and epoch_cur==0:

                cur=images[0].detach()
                # cur=torch.unsqueeze(cur, 0)
                # cur=F.interpolate(cur,size=(55,55),mode='bilinear',align_corners=False)
                # cur=F.interpolate(cur,size=(55,55),mode='nearest')
                cur=cur.squeeze(0).permute((1,2,0))
                cur=cur.detach().cpu().numpy()
                cur=cur.squeeze()
                cur = np.asarray(cur,dtype=np.uint8)
                
                # plt.imshow(cur)
                # plt.axis('off')
                # plt.savefig('patch_for_vis.jpg')
                
                plt.imsave(os.path.join(self.debug_info,'attention_sum','patch_img_batch'+str(batch_idx)+'_epoch_'+str(epoch_cur)+'.jpg'),cur)
                
        
        

        
        if status=='step1' or status=='step2':
            tokens= self.patchify(images)
        
            out = torch.stack([torch.vstack((self.class_token,tokens[i])) for i in range(len(tokens))])
            # out = out + torch.tensor(0.1)*self.pos_embed
            attention_all=[]
            # attention_all1=[]
            for block in self.blocks:
                out,attention = block(out)
                if self.return_attention:
                    attention_all.append(attention)
                    # attention1=self.avg_pool(attention)
                    # attention_all1.append(attention1)
    
            out1 = self.gene_expression_many_to_one(out[:,:self.n_genes]) #(N,,hidden_dim) -> (N,,M)
            out1=out1.squeeze()

            if status=='step1':
                return out1,None


            ###Step 2 only
            ###Get proposals - Find ROIs
            attention_all = torch.stack(attention_all)
            # attention_all1 = torch.stack(attention_all1)
            object_labeled_mask,labels,attention_avg,guided_object_prev_filtering=self.AttentionGradientFlowTracking(attention_all,bin_size=1,threshold=self.threshold,epoch_cur=epoch_cur,batch_idx=batch_idx,images=images)
            # object_labeled_mask,labels=self.AttentionGradientFlowTracking(attention_all1,bin_size=1,threshold=self.threshold,epoch_cur=epoch_cur,batch_idx=batch_idx,images=images)                        
            # attention_img=attention_img.flatten(start_dim=2,end_dim=3)
            # attention_img=attention_img.squeeze()###Should be changed if more attention layers
            return out1,object_labeled_mask,attention_avg,guided_object_prev_filtering

                

            if status=='step3':
                out= out.sum(dim=1)
            return out
##########################!!!
class GeneTransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio = 4):
        super(GeneTransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadSelfAttention(hidden_d,n_heads)
        
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d, hidden_d),
            # nn.Linear(mlp_ratio*hidden_d, 117),
            # nn.ReLU()
            
            )
    
    def forward(self, x):
        out, attention =self.mhsa(self.norm1(x))
        out = x+ out
        out = out+self.mlp(self.norm2(out))
        return out, attention




def clamp_regressor_nonreg(m):
    with torch.no_grad():
        for module in m.module.regressor.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.clamp_(min=0.0)
                if module.bias is not None:
                    module.bias.data.clamp_(min=0.0)
                
                
                
#################################
class GeneBlock(nn.Module):
    def __init__(self,dim_in=1,dim_out=1):
        super(GeneBlock,self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim_in,dim_out),
            # nn.Conv1d(dim_in, dim_out,5, padding=2),
            # nn.ELU()
            )
    
    def forward(self,x):
        return self.net(x)

class AnnBlock(nn.Module):
    def __init__(self,dim_in=1,dim_out=1):
        super(AnnBlock,self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim_in,dim_out),
            nn.ELU()
            )
    
    def forward(self,x):
        return self.net(x)

class RegressionTransformer2(nn.Module):
    def __init__(self,num_label_channel=None,nochw=(1,1,1,28,28),spot_nchw=(1,1,28,28),patch_size=7,n_batches=10, n_blocks = 2, input_d = 8, n_heads =2,n_genes=117, out_d=10,device='cpu',threshold=0,max_num_cells=10,return_attention=True,debug_info=False,seq=None,target_layer_name='regressor'):
        super(RegressionTransformer2,self).__init__()
        self.nochw =nochw
        self.n_batches = n_batches
        self.input_d = input_d
        self.patch_size=patch_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_genes = n_genes
        self.device=device
        self.return_attention = return_attention
        self.debug_info = debug_info
        self.hidden_d_1st_conv = 25 ##1152
        self.max_num_cells = max_num_cells
        self.spot_nchw= spot_nchw
        self.spot_img_embedding_dim = 3
        self.num_label_channel =num_label_channel
        # self.hidden_d_1st_conv=147
        
        self.hidden_d= self.hidden_d_1st_conv*self.nochw[3]*self.nochw[4]
        self.threshold = threshold
        ###Learnable class token
        # self.gene_token = nn.Parameter(torch.rand(self.n_genes,self.hidden_d_1st_conv))
        self.gene_token = nn.Parameter(torch.rand(1,self.hidden_d_1st_conv))
        ###Positional embedding
        # self.patchify = PatchEmbedding(self.spot_nchw,patch_size,self.spot_img_embedding_dim)
        self.patchify = PatchEmbedding(self.spot_nchw,5,self.spot_img_embedding_dim)
        ###Transformer encoder blocks

        self.Emphsis = nn.Sequential(
            nn.Conv2d(self.nochw[2], self.hidden_d_1st_conv, kernel_size=3,stride=1),
            nn.ReLU(),
            
            )


        weight_ones = torch.ones(1,max_num_cells)
        self.register_buffer("weight_ones", weight_ones)
        bias_ones = torch.zeros(1)
        self.register_buffer("bias_ones", bias_ones)

        self.spot_dist = nn.Linear(1,self.hidden_d_1st_conv)
        self.regressor=nn.Sequential(
            nn.Linear(625,1),
            nn.ReLU(),
            )
        self.relu =nn.ReLU()


        self.regressor=nn.ModuleList([GeneBlock(625,1) for _ in range(self.n_genes)])

        
        self.global_embedding = nn.Sequential(
            nn.Linear(75,10),
            nn.ELU()
            
            )
        self.temporal = nn.Sequential(
            nn.Linear(50,25),
            nn.ELU(),
            nn.Linear(25,1),
            nn.ELU(),
            # nn.Linear(50,5),
            
            
            )
        self.temp=nn.Linear(625,10)
        
        
        ###Method 3
        self.part_spot = nn.Sequential(
            nn.Linear(10,1),
            nn.ELU(),
            # nn.Linear(50,5),
            )


        
        ###Method 4
        self.part_spot = nn.Sequential(
            nn.Linear(11,1),
            nn.ELU(),
            # nn.Linear(50,5),
            )        

        self.regressor=nn.Sequential(
            nn.Linear(625,1),
            nn.ReLU(),
            )
        self.relu =nn.ReLU()


        self.regressor=nn.ModuleList([GeneBlock(625,1) for _ in range(self.n_genes)])

        
        self.global_embedding = nn.Sequential(
            nn.Linear(75,1),
            nn.ELU()
            
            )

        
        # Build per-sequence learning rate vector (shape: (1, M, 1))
    def freeze_genes(self,gene_indices,emphasis_frozen_bool):
        if emphasis_frozen_bool:
            for p in self.Emphsis.parameters():
                p.requires_grad= False
        for i in gene_indices:
            for p in self.regressor[i].parameters():
                p.requires_grad= False
            # self.frozen[i]=True



        

    def forward(self, images,spot_images,gene_expression,cur_human_annotation=None,cur_true_detected_cell_mask=None,epoch_cur=-1,batch_idx=None,mode='step4'):
            '''
            (Batch, Genes, Objects, Hidden Dim)
            
            '''
  

            ###Method 5 - current version
            tokens=[]
            for i in range(len(images)):
                tokens.append(self.Emphsis(images[i]))
            tokens=torch.stack(tokens)
            out=tokens.flatten(start_dim=2) #(N, NumObj, W'H'D')
            

            
            out = [self.regressor[i](out) for i in range(self.n_genes)]
            # out = torch.stack(out,dim=2).squeeze() ##Worked well
            out = torch.stack(out,dim=2)
            # out = self.relu(out)
            
            spot_images=self.patchify(spot_images)
            spot_images = self.global_embedding(spot_images.flatten(start_dim=1))
            out1 = spot_images.unsqueeze(1).unsqueeze(1).repeat(1,out.size(1),out.size(2),1)

            
            # out1 = torch.cat([out,out1],dim=3)
            

            # cell = self.part_spot(out1)
            # cell = out
            cell = out+out1##Worked well
            # cell = cell.squeeze()
            # cell = out
            cell = self.relu(cell)
            # cell = out*out1
            
            cell = cell.squeeze()
            if out.size(0)==1:
                cell = cell.unsqueeze(0)
            
            spot = cell.permute(0,2,1)
            spot = F.linear(spot,self.weight_ones,self.bias_ones)
            spot = spot.permute(0,2,1).squeeze()
            if out.size(0)==1:
                spot = spot.unsqueeze(0)
      
            # out1 = out1.permute(0,2,1)
            # out1 = F.linear(out1,self.weight_ones,self.bias_ones)
            # out1 = out1.permute(0,2,1).squeeze()
            # if out.size(0)==1:
            #     out1 = out1.unsqueeze(0)
            
            return spot, cell, out1.squeeze()      
            print()


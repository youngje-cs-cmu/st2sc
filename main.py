#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:19:02 2024

@author: yjadvance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as torchvision_models
import torch.utils.data as data_utils
from models import VGT, RegressionTransformer2
import torchvision
import skimage
import os
import copy
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pickle

import sys
args= sys.argv

# num_epochs = int(args[1])
# gene_img_save_path = args[2]
# unique_id = args[3]

# print('args')
# print('num_epochs: ',num_epochs)
# print('gene_img_save_path: ',gene_img_save_path)
# print('unique_id: ',unique_id)
#####
###Transformer params
num_epochs=20
unique_id='test'
gene_img_save_path='./AGFT_pseudo_xenium_'+unique_id
first_step = True ##'True','False'
num_epochs_2nd=10 
num_batch=5
patch_size=55

n_heads=3
n_hidden = 67*n_heads
n_input_d = 24 
n_hidden=12


region_of_interest_format = 'img' ##'img', 'attention'
num_test_spot=100

dataset='sim' 
gene_exp='true' ##'random'
#####
os.makedirs(gene_img_save_path,exist_ok=True)
os.makedirs(os.path.join(gene_img_save_path,'attention_sum'),exist_ok=True)

if dataset=='sim':
    num_batch=5
    threshold=2
    n_genes=313
    image_shape_spot=(num_batch,3,100,100)
    image_shape_spot=(num_batch,3,55,55)
    data_path='./test_data/generated_data.h5'
    data=h5py.File(data_path, "r")
    
    
    counts = scipy.sparse.csr_matrix(
        (
            data["counts"]["data"], 
            data["counts"]["indices"],
            data["counts"]["indptr"],
        ),
        shape=(
            len(data["counts"]["index"]),
            len(data["counts"]["columns"]),
        ),
    )
    
    counts=counts.toarray()

    image=data['image'][()]
    label=data['label'][()]
    
    
    grid=100
    grid_sub=int(grid/10)
    img_whole_spot_level=[]
    for i in range(10):
        for j in range(10):
            coords=image[i*grid:(i+1)*grid,j*grid:(j+1)*grid,:]
            coords=skimage.transform.resize(coords,(patch_size,patch_size,3))
            img_whole_spot_level.append(coords)
    train_x_spot=np.asarray(img_whole_spot_level)
        
    


 


#####Case we want to use whole spot image
train_x=torch.from_numpy(train_x_spot).float()
train_y=torch.from_numpy(counts).float()
train_x=train_x.permute(0,3,1,2)

##### MODEL
num_genes=train_y.shape[1]
device='cuda:0'#'cuda:0'

train_y=train_y.to(device)

model = VGT(nchw=image_shape_spot,n_batches=num_batch,patch_size=patch_size, n_blocks = 1, hidden_d =n_hidden, n_genes=num_genes, n_heads =n_heads,threshold=threshold,device=device,debug_info=gene_img_save_path)

optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion = nn.MSELoss()

###Load saved model
model=nn.DataParallel(model)
model.to(device)




###Train
train = data_utils.TensorDataset(train_x,train_y)
train_loader = data_utils.DataLoader(train, batch_size=num_batch)   

test_loss=[]
test_outputs=[]
if first_step:
    print('Entering training session')
    training_time = time.time()
    for i in range(num_epochs):
        cur_test_loss=0
        for idx, (x,y) in enumerate(train_loader):
        # Train(model,train_x,train_y,criterion,optimizer)
        
            model.train()
            optimizer.zero_grad()
            model_output,attention=model(x,status='step1',epoch_cur=i,batch_idx=idx)
            # model_output=model(train_x_vit)
            loss = criterion(model_output,y)
            loss.backward()
            optimizer.step()
        
        
            model.eval()
            with torch.no_grad():
                model_output,attention=model(x,status='step1',epoch_cur=i,batch_idx=idx)
                cur_test_loss+=criterion(model_output,y).item()
                test_loss.append(cur_test_loss)
                # test_outputs.append(test)
        print('Epoch: '+str(i)+', loss: '+str(cur_test_loss))
    
    
    ###Get the proposals from AGFT

if first_step:
    print('Get the interesting points')

    proposals=[]
    attention_all=[]
    images_all = []
    guided_object_prev_filtering_all=[]
    model.eval()
    for i in range(1):
        for idx, (x,y) in enumerate(train_loader):

            model_output,object_labels,attention_avg,guided_object_prev_filtering=model(x,status='step2',epoch_cur=i,batch_idx=idx)
            attention_all.append(attention_avg.detach().cpu())
            proposals.append(object_labels.detach().cpu())
            guided_object_prev_filtering_all.append(guided_object_prev_filtering.detach().cpu())
            images_all.append(x)
            
            if idx%50==0:
                print(idx,' th idx completed in Getting the interesting points part')
            

    proposal_acquisition_time=time.time()
    print('Proposal obtined with time: ',proposal_acquisition_time-training_time)
    
 
    
    
    '''
    Currently, we have proposals in the one image.
    
    Now split for each proposal. 
    '''
    size_filter=20
    proposal_size=(7,7)
    max_num_cells=0
    proposal_group=[]
    cell_to_index_all=[]
    for i in range(len(proposals)):
        for j in range(len(proposals[i])):
            ###0th index is labels
            cur_image = proposals[i][j]
            ###We have number of objects in num_labels_all
            current_proposals=[]
            cell_to_index_key=0
            cell_to_index_dict={}
            # ids=[]
            num_labels_all = np.unique(cur_image)
            num_labels_all = num_labels_all[np.nonzero(num_labels_all)]
            preprocessed_output_img=torch.zeros(cur_image.shape)
            for cur_label in num_labels_all:
                cur_label_image = (cur_image==cur_label)
                cur_label_image_nonzero = cur_label_image.nonzero(as_tuple=False)
                # cur_label_image
                # print()
                if cur_label_image_nonzero.size(0)==0:
                    continue
                y_min,x_min = cur_label_image_nonzero.min(dim=0).values
                y_max,x_max = cur_label_image_nonzero.max(dim=0).values
                # region_of_interest=proposals[i][j]
                
                if (y_max-y_min)*(x_max-x_min)>cur_label_image.size(0)*cur_label_image.size(1)*0.8:
                    continue
                
                if torch.sum(cur_label_image)<size_filter:
                    continue
                
                ####Save filtered image here
                preprocessed_output_img+=cur_label_image
                
                
                ####
                if region_of_interest_format == 'img':
                    region_of_interest=images_all[i][j]
                # region_of_interest=region_of_interest[:,y_min:y_max+1,x_min:x_max+1]
                # region_of_interest=region_of_interest.unsqueeze(0)
    
                elif region_of_interest_format == 'attention':
                    region_of_interest=images_all[i][j]
                    # region_of_interest=region_of_interest[:,y_min:y_max+1,x_min:x_max+1]
                    # region_of_interest=region_of_interest.unsqueeze(0)
    
                region_of_interest=region_of_interest[:,y_min:y_max+1,x_min:x_max+1]
                region_of_interest=region_of_interest.unsqueeze(0)
                region_of_interest=F.interpolate(region_of_interest,size=proposal_size,mode='nearest')            
                region_of_interest=region_of_interest.squeeze()
                
                current_proposals.append(region_of_interest)
                cell_to_index_dict[cur_label]=cell_to_index_key
                cell_to_index_key+=1
    
            cell_to_index_all.append(cell_to_index_dict)
            proposal_group.append(current_proposals)
            
            preprocessed_output_img = preprocessed_output_img*cur_image
            if i in [0,1,2,3,4,5,6,7,8,9,10] and j==0:
                plt.imsave(os.path.join(gene_img_save_path,'attention_sum','postprocessed'+str(i)+'_epoch_'+str(j)+'.jpg'),preprocessed_output_img)
                torch.save(preprocessed_output_img,os.path.join(gene_img_save_path,'attention_sum','postprocessed'+str(i)+'_epoch_'+str(j)+'.pt'))
                
            
    
            if len(current_proposals)>max_num_cells:
                max_num_cells=len(current_proposals)
        
        if i%50==0:
            print(i,' th proposal completed in Getting the first processing part')

    ###
    
    '''
    proposal_group = (Cells per region, H', W')
    
    '''
    print('First processing done: ',proposal_acquisition_time-time.time())
            
    proposal_group1=[]        
    for i in range(len(proposal_group)):
        try:
            cur_proposal_group=torch.stack(proposal_group[i])
        except:
            cur_proposal_group=torch.zeros((3,proposal_size[0],proposal_size[1])).unsqueeze(0)
        if len(list(cur_proposal_group.size()))==3:
            cur_proposal_group=torch.unsqueeze(cur_proposal_group, 1)
        adding_size=list(cur_proposal_group.size())
        adding_size[0]=max_num_cells-adding_size[0]
        cur_proposal_group=torch.vstack((cur_proposal_group,torch.zeros(adding_size)))
        proposal_group1.append(cur_proposal_group)
    
    proposal_group1=torch.stack(proposal_group1)




    proposals_group_format=[]
    for i in range(len(proposals)):
        for j in range(len(proposals[i])):
            proposals_group_format.append(proposals[i][j])


with open('cell_to_index_all.pkl', "rb") as f:
    cell_to_index_all = pickle.load(f)
with open('proposals.pkl', "rb") as f:
    proposals = pickle.load(f)
with open('proposal_group.pkl', "rb") as f:
    proposal_group = pickle.load(f)
with open('proposal_group1.pkl', "rb") as f:
    proposal_group1 = pickle.load(f)
with open('max_num_cells.txt', "r") as f:
    max_num_cells = int(f.read())
max_num_cells = 12


###TEMPORAL: Make a mask label
true_detected_cell_mask = np.zeros((len(cell_to_index_all),max_num_cells))
for i in range(len(cell_to_index_all)):
    true_detected_cell_mask[i,:len(cell_to_index_all[i])]=1
true_detected_cell_mask = torch.from_numpy(true_detected_cell_mask)



train_x_spot=torch.from_numpy(train_x_spot).float()
train_x_spot=train_x_spot.permute(0,3,1,2)

image_shape=proposal_group1.size()
train = data_utils.TensorDataset(proposal_group1, train_y, train_x_spot, true_detected_cell_mask)
train_loader = data_utils.DataLoader(train, batch_size=num_batch)

model_reg = RegressionTransformer2(nochw=image_shape,spot_nchw=image_shape_spot,n_batches=num_batch,patch_size=patch_size, n_blocks = 2,  input_d =n_input_d, n_genes=num_genes, n_heads =n_heads,threshold=threshold,max_num_cells=max_num_cells,device=device,debug_info=gene_img_save_path)
model_reg.to(device)



###!!!
def EvaluatePerGene(output,groundtruth):
    res = ((output-groundtruth)**2).sum(dim=0)
    return res
def heteroscedastic_gaussian_nll(y_pred_mu, y_pred_logvar, y_true, reduction='mean'):
    """
    y_pred_mu: (N, T) predicted mean
    y_pred_logvar: (N, T) predicted log variance (log σ²)
    y_true: (N, T) true targets
    """
    # compute per-target NLL
    loss = 0.5 * (torch.exp(-y_pred_logvar) * (y_true - y_pred_mu)**2 + y_pred_logvar)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction
class NBLoss(nn.Module):
    def forward(self, mu, log_theta, target):
        theta = torch.exp(log_theta)
        eps = 1e-8
        t1 = torch.lgamma(theta + eps) + torch.lgamma(target + 1.0) - torch.lgamma(target + theta + eps)
        t2 = (theta + target) * torch.log1p(mu / (theta + eps)) + target * (torch.log(theta + eps) - torch.log(mu + eps))
        loss = t1 + t2
        return torch.mean(loss)    

def pcc_loss(y_pred, y_true, eps=1e-8):
    # mean over batch
    x = y_pred - y_pred.mean(dim=0, keepdim=True)
    y = y_true - y_true.mean(dim=0, keepdim=True)
    cov = (x * y).sum(dim=0)
    corr = cov / (torch.sqrt((x**2).sum(dim=0) * (y**2).sum(dim=0)) + eps)
    return corr.mean()  # mean across targets

reg_lr = 0.003
emphasis_frozen_bool=False
print('Run regression per object')
criterion = nn.HuberLoss(reduction='none')

optimizer=optim.Adam(model_reg.parameters(),lr=reg_lr)


def soft_rank(x, tau=1.0):
    x = x.unsqueeze(-1)
    diff = x - x.transpose(-1, -2)
    P = torch.sigmoid(-diff / tau)
    return P.sum(dim=-1)


def spearman_loss(x, y, tau=0.1):
    rx = soft_rank(x, tau)
    ry = soft_rank(y, tau)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return 1 - (rx * ry).mean()  # (1 - rho) can be used as loss
def spearman_corr(x, y, tau=1.0):
    """
    Differentiable Spearman correlation between x and y.
    """
    rx = soft_rank(x, tau)
    ry = soft_rank(y, tau)
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return (rx * ry).mean()

def safe_spearman_reg(preds, targets, weight=0.05, tau=0.5, eps=1e-8):
    reg = 0.0
    valid_count = 0
    for i in range(preds.shape[1]):
        x = preds[:, i]
        y = targets[:, i]

        # Check for degenerate variance
        if x.std() < eps or y.std() < eps:
            continue

        rho = spearman_corr(x, y, tau=tau)
        reg += (1 - rho)
        valid_count += 1

    if valid_count > 0:
        reg /= valid_count
    else:
        reg = torch.tensor(0.0, device=preds.device)
    return weight * reg



test_loss=[]
mse_per_gene_all=[]
evaluate_res_mask = torch.ones(num_genes).float().to(device)
evaluate_res_mask_set=set()
for i in range(num_epochs_2nd):
    mse_per_gene=[]

    
    cur_test_loss=0
    cur_pcc=0
    batch_tracker=0
    
    for idx, (x,y,spot_img,cur_true_detected_cell_mask) in enumerate(train_loader):
        
        x=x.to(device)
        y=y.to(device)
        spot_img=spot_img.to(device)
        cur_true_detected_cell_mask=cur_true_detected_cell_mask.to(device)

        optimizer.zero_grad()
        model_reg.train()
        
        model_output, out_reg,out_var=model_reg(x,spot_img,y,cur_true_detected_cell_mask=cur_true_detected_cell_mask,epoch_cur=i,batch_idx=idx,mode='step3')
        
        batch_tracker+=1
        loss = criterion(model_output,y)

        loss = loss.mean()
        
        loss.backward()
        
        optimizer.step()
        

    
    
        model.eval()
        with torch.no_grad():
            # model_output, cell, _=model_reg(x, spot_img, y,cur_human_annotation=cur_human_annotation,cur_true_detected_cell_mask=cur_true_detected_cell_mask, epoch_cur=i,batch_idx=idx,mode='step3')
            model_output, cell,out_var=model_reg(x, spot_img, y,cur_true_detected_cell_mask=cur_true_detected_cell_mask, epoch_cur=i,batch_idx=idx,mode='step3')
            cur_test_loss+=criterion(model_output,y).mean().item()
    test_loss.append(cur_test_loss)
    print('Epoch: '+str(i)+', loss: '+str(cur_test_loss))



print('Evaluate')

object_level_expressions_record_all_temp=[]
model.eval()
for i in range(1):
    for idx, (x,y,spot_img, cur_true_detected_cell_mask) in enumerate(train_loader):

        x=x.to(device)
        y=y.to(device)
        spot_img=spot_img.to(device)
        cur_true_detected_cell_mask=cur_true_detected_cell_mask.to(device)
        
        model_output, cell,out_var=model_reg(x, spot_img, y,cur_true_detected_cell_mask=cur_true_detected_cell_mask, epoch_cur=i,batch_idx=idx)
                
        object_level_expressions_record_all_temp.append(cell)

object_level_expressions_record_all=[]
for i in range(len(object_level_expressions_record_all_temp)):
    for j in range(len(object_level_expressions_record_all_temp[i])):
        object_level_expressions_record_all.append(object_level_expressions_record_all_temp[i][j])

object_level_expressions_record_all=torch.stack(object_level_expressions_record_all)



print('Done')


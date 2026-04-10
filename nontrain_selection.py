import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import time
from pathlib import Path

import open_clip
from open_clip import tokenizer, tokenize
from data import get_data, get_data_val
from text_preprocessing import text_preprocessing


def _extract_urls(batch_meta):
    """Normalize metadata batch into a list of URL strings."""
    if isinstance(batch_meta, dict):
        urls = batch_meta.get("url", [])
        if isinstance(urls, (list, tuple, np.ndarray)):
            return [str(u) for u in urls]
        return [str(urls)]

    if isinstance(batch_meta, (list, tuple, np.ndarray)):
        urls = []
        for item in batch_meta:
            if isinstance(item, dict):
                urls.append(str(item.get("url", "")))
            else:
                urls.append(str(item))
        return urls

    return [str(batch_meta)]


def _load_overlap_array(filename):
    """Load overlap metadata from common project locations.

    If unavailable, returns an empty array so the caller can continue
    without overlap-based filtering.
    """
    candidate_paths = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / "data" / filename,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return np.load(candidate)

    searched = ", ".join(str(p) for p in candidate_paths)
    print(f"[WARN] overlap metadata not found: {filename}. Continue without overlap filtering. searched={searched}")
    return np.array([], dtype=str)


def select_nontrain(args, target_model, preprocess_train, preprocess_val, device, length):

    selected_nt_txt = []
    selected_nt_url = []
    selected_nt_cs_lst_tar = []
    selected_nt_feat_lst_tar = []
        
    args.val_data = args.val_data_nontrain_1
    args.val_num_samples = args.val_num_samples_nontrain_1
    valloader = get_data_val(args, (preprocess_val, preprocess_val))
    valloader = valloader.dataloader    

    ##################################################    
    ### Nontrain data from valloader
    start_time = time.time()

    CC3M_LAION_commonset = _load_overlap_array('CC3M_LAION_commonset.npy')
    CC3M_LAION_unqiue_commonset = _load_overlap_array('CC3M_LAION_unqiue_commonset.npy')
    CC3M_LAION_url_commonset = _load_overlap_array('CC3M_LAION_url_commonset.npy')
    
    cnt_nontrain = 0
    for i, batch in enumerate( valloader ): 
        non_train_text = [text_preprocessing(q) for q in batch[1]]            
        non_train_url = _extract_urls(batch[2])

        common = np.intersect1d(np.array(non_train_text), CC3M_LAION_commonset)
        x_ind = np.where(np.isin(np.array(non_train_text), common))[0]
        common_2 = np.intersect1d(np.array(non_train_url), CC3M_LAION_url_commonset)
        x_ind_2nd = np.where(np.isin(np.array(non_train_url), common_2))[0]
        combined_x_ind = np.union1d(x_ind, x_ind_2nd)
        
        sampled_ind = np.random.choice(np.arange(len(batch[1])), int(len(batch[1])*0.05), replace=False) 
        selected_ind = np.setdiff1d(sampled_ind, combined_x_ind)
            
        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        cnt_nontrain += len(images)

        images = images.to(device)
        texts = texts.to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)
        cs_2 = torch.diagonal( image_features2@text_features2.T )
        
        selected_nt_url.extend( np.array(non_train_url)[selected_ind] )
        selected_nt_txt.extend( np.array(non_train_text)[selected_ind] ) 
        selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )
        
        if cnt_nontrain >= length:
            break
    
    ##################################################
    ## nontrain data | collect the data from the second val loader : CC12M

    start_time = time.time()

    CC12M_LAION_commonset = _load_overlap_array('CC12M_LAION_commonset.npy')
    CC12M_LAION_unqiue_commonset = _load_overlap_array('CC12M_LAION_unqiue_commonset.npy')
    CC12M_LAION_url_commonset = _load_overlap_array('CC12M_LAION_url_commonset.npy')
    
    args.val_data = args.val_data_nontrain_2
    args.val_num_samples = args.val_num_samples_nontrain_2
    cc12m_valoader = get_data_val(args, (preprocess_val, preprocess_val))
    cc12m_valoader = cc12m_valoader.dataloader

    ## nontrain data 
    cnt_nontrain = 0
    for i, batch in enumerate( cc12m_valoader ): 

        non_train_text = [text_preprocessing(q) for q in batch[1]]            
        non_train_url = _extract_urls(batch[2])

        common = np.intersect1d(np.array(non_train_text), CC12M_LAION_commonset)
        x_ind = np.where(np.isin(np.array(non_train_text), common))[0]
        common_2 = np.intersect1d(np.array(non_train_url), CC12M_LAION_url_commonset)
        x_ind_2nd = np.where(np.isin(np.array(non_train_url), common_2))[0]
        combined_x_ind = np.union1d(x_ind, x_ind_2nd)

        sampled_ind = np.random.choice(np.arange(len(batch[1])), int(len(batch[1])*0.05), replace=False)
        selected_ind = np.setdiff1d(sampled_ind, combined_x_ind)
        
        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        cnt_nontrain += len(images)
        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal(image_features2@text_features2.T)
        
        selected_nt_url.extend( np.array(non_train_url)[selected_ind] )
        selected_nt_txt.extend( np.array(non_train_text)[selected_ind] )
        selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )
        
        if cnt_nontrain >= length:
            break
 
    ################################################## 
    ## nontrain data | collect the data from the second val loader : MSCOCO
    
    args.val_data = args.val_data_nontrain_3
    args.val_num_samples = args.val_num_samples_nontrain_3
    mscoco_valoader =  get_data_val(args, (preprocess_val, preprocess_val))
    mscoco_valoader = mscoco_valoader.dataloader

    start_time = time.time()
    MSCOCO_LAION_commonset = _load_overlap_array('MSCOCO_LAION_commonset.npy')
    MSCOCO_LAION_unqiue_commonset = _load_overlap_array('MSCOCO_LAION_unqiue_commonset.npy')

    ## nontrain data 
    cnt_nontrain = 0
    for i, batch in enumerate( mscoco_valoader ): 
        
        non_train_text = [text_preprocessing(q) for q in batch[1]]   
        non_train_url = _extract_urls(batch[2])

        common = np.intersect1d(np.array(non_train_text), MSCOCO_LAION_commonset)
        x_ind = np.where(np.isin(np.array(non_train_text), common))[0]
        combined_x_ind = x_ind
        
        sampled_ind = np.random.choice(np.arange(len(batch[1])), int(len(batch[1])*0.05), replace=False)
        selected_ind = np.setdiff1d(sampled_ind, combined_x_ind)

        images, texts = batch[0][selected_ind], tokenize(batch[1])[selected_ind]
        cnt_nontrain += len(images)

        images = images.to(device)
        texts = texts.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features2, text_features2, logit_scale2 = target_model(images, texts)  
        cs_2 = torch.diagonal( image_features2@text_features2.T )

        selected_nt_url.extend( np.array(non_train_url)[selected_ind] )
        selected_nt_txt.extend( np.array(non_train_text)[selected_ind] )
        selected_nt_cs_lst_tar.extend( cs_2.detach().cpu().numpy() ) 
        selected_nt_feat_lst_tar.extend( torch.cat([image_features2, text_features2], dim=1).detach().cpu() )

        if cnt_nontrain >= length:
            break

    return selected_nt_txt, selected_nt_url, selected_nt_cs_lst_tar, selected_nt_feat_lst_tar

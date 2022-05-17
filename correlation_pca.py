#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:06:09 2021

@author: qwer7892228
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import read as rd
from scipy.stats import pearsonr
import pca_br as pb
import pca_sill as ps

def BrSillass_df(BR_pca,SILL_pca):

    BR_pca = BR_pca.T
    SILL_pca = SILL_pca.T
        
    BR_pca.reset_index(inplace=True)
    SILL_pca.reset_index(inplace=True)
        
    SILL_pca = SILL_pca.rename(columns={"學號 (Student ID)":"username"})
    ass_df = pd.merge(SILL_pca,BR_pca,how = 'inner')
    ass_df.set_index('username',inplace=True)
    
    rho = np.abs(ass_df.corr())
    pval = ass_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
    
    brsill_corr = rho.round(2)
    brsill_corr = brsill_corr.iloc[SILL_pca.shape[1]-1:,:SILL_pca.shape[1]-1]
    rho_str = rho.applymap(lambda x:"%.2f" % x)
    brsill_corr_p = rho_str + p
    brsill_corr_p = brsill_corr_p.iloc[SILL_pca.shape[1]-1:,:SILL_pca.shape[1]-1]
    brsill_corr = rho.iloc[SILL_pca.shape[1]-1:,:SILL_pca.shape[1]-1]
    
    plt.figure(figsize = (30,10))
    ax = sns.heatmap(brsill_corr, annot = brsill_corr_p, cmap="BuPu", annot_kws={"size":15},fmt = '')
    ax.yaxis.set_tick_params(labelsize=12,rotation = 0)
    plt.savefig('./images/brsill_pca_corr_heatmap.png', dpi=300)

    brsill_corr.to_csv('./outputs/brsill_pca_corr.csv', index=True,encoding='big5')
    
    return(ass_df,brsill_corr)

ex_sill,SILL_pca = ps.pca_sill()
ex_br,BR_pca,BRmg_df = pb.pca_br()
ass_df,brsill_corr = BrSillass_df(BR_pca,SILL_pca)
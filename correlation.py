#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:34:02 2021

@author: qwer7892228
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import read as rd
import seaborn as sns
from scipy.stats import pearsonr


    
def BrSillass_df(df_ab):
    
    df_ab,df_ABex = rd.handle_df_ab()
    marker,bookmark,memo,BRmg_df,nor_df = rd.br_read(df_ABex)
    
    df_ab.reset_index(inplace=True)
    df_ab = df_ab.rename(columns={"學號 (Student ID)":"username"})

    brsill_corr = pd.DataFrame()
    brsill_corr_p = pd.DataFrame()
        
    for z in (marker,bookmark,memo):
        z = z.rename(columns={"編碼名稱":"username"})
        ass_df = pd.merge(df_ab,z,how = 'inner')
        ass_df.set_index('username',inplace=True)
            
        rho = np.abs(ass_df.corr())
        pval = ass_df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
            
        corr = rho.round(2)
        corr = corr.iloc[df_ab.shape[1]-1:,:df_ab.shape[1]-1]
        rho_str = rho.applymap(lambda x:"%.2f" % x)
        corr_p= rho_str + p
        corr_p = corr_p.iloc[df_ab.shape[1]-1:,:df_ab.shape[1]-1]
            
       
        brsill_corr = brsill_corr.append(corr)
        brsill_corr_p = brsill_corr_p.append(corr_p)
            
    ### Show Chinese
    plt.rcParams['font.sans-serif']=['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False 
        
        
    plt.figure(figsize = (65,10))
    ax = sns.heatmap(brsill_corr, annot=brsill_corr_p, cmap="BuPu", annot_kws={"size":15},fmt='')
    ax.yaxis.set_tick_params(labelsize=12,rotation = 0)
        
    ### Set y-axis label
    plt.savefig('./images/brsill_corr_heatmap.png', dpi=300)
        
    brsill_corr.to_csv('./outputs/brsill_corr.csv', index=True,encoding='big5')
    
df_ab,df_ABex = rd.handle_df_ab()
BrSillass_df(df_ab)
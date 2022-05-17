#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:13:51 2021

@author: qwer7892228
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import stats
from sklearn.decomposition import PCA
import read as rd

def pca_sill():
    
    df_ab,df_ABex = rd.handle_df_ab()
    
    
    df_ABex = df_ABex [['學號','學期成績']]
    df_ABex = df_ABex.rename(columns={'學號':"學號 (Student ID)"})
    df_ab.reset_index(inplace=True)
    
    nor_df = pd.merge(df_ab,df_ABex)
    
    df_ab.set_index('學號 (Student ID)',inplace=True)
    #standardization
    nor = normalization(df_ab)
    
    pca=PCA(n_components=24,random_state=9527)
    
    L_sk = pca.fit_transform(nor).T
    #set up dataframe index,columns
    
    ex = np.array(pca.components_)
    ex_df = pd.DataFrame(ex, columns=df_ab.columns[0:])
    
    plt.figure()
    cum_explained= np.cumsum(pca.explained_variance_ratio_)
    plt.text(27,0.64,'%.2f'%cum_explained[23], ha='center',va='bottom', wrap=True)
    plt.plot([24, 27], [0.9,0.7], c='b', linestyle='-')
    plt.vlines(24,0,1, linestyle="dashed")
    index = np.arange(cum_explained.shape[0])+1
    plt.plot(index,cum_explained,marker = "o") 
    x = [1,10,20,24,30,40,50]
    plt.xticks(x)
    plt.ylim(0,None)
    plt.xlabel('# principal components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('./images/sill_pca.png',dpi = 300)
    
    #count p-value
    ex_df = ex_df.round(2)
    ex_df_p = ex_df.applymap(lambda x:"%.2f" % x)
    
    for i in range(0,L_sk.shape[0],1):
        
        r_value,p_value=stats.pearsonr(L_sk[i],nor_df['學期成績'])
        
        if 0.1 <= p_value < 0.05:
            p_value = '%.2f'%r_value+'*'
        elif 0.001 <= p_value < 0.01:
            p_value = '%.2f'%r_value+'*'
        elif p_value < 0.001:
            p_value = '%.2f'%r_value+'***'
        else:
            p_value = '%.2f'%r_value
        
        
        ex_df.loc[i,'p_value'] = r_value
        ex_df_p.loc[i,'p_value'] = p_value
    
    

    
    plt.figure(figsize = (60,10))
    
    ax = sns.heatmap(ex_df, annot = ex_df_p, cmap="BuPu", annot_kws={"size":15},fmt = '')
    ax.yaxis.set_tick_params(labelsize=12,rotation = 0)
    
    ### Set y-axis label        
    plt.savefig('./images/sill_pca_heatmap.png', dpi=300)
    
    L_sk = pd.DataFrame(L_sk, columns=df_ab.index)
    L_sk.index = [f"SILL{d}" for d in range(1,L_sk.shape[0]+1)]
    ex_df.to_csv('./outputs/sill_pca.csv', index=True, encoding='big5')
   
    return(ex,L_sk)

def normalization(df_ab):
    from sklearn import preprocessing
    nor = preprocessing.scale(df_ab)
    return(nor)

ex_sill,SILL_pca = pca_sill()
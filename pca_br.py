#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:47:11 2021

@author: qwer7892228
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import stats
from sklearn.decomposition import PCA
import read as rd



def pca_br():
    
    df_ab,df_ABex = rd.handle_df_ab()
    
    marker,bookmark,memo,BRmg_df,nor_df = rd.br_read(df_ABex)
    #standardization
    nor = normalization(BRmg_df)
    
    pca=PCA(n_components=9,random_state=9527)
    L_sk = pca.fit_transform(nor).T
    
    
    ex = np.array(pca.components_)
    ex_df = pd.DataFrame(ex,columns=BRmg_df.columns)
    cum_explained= np.cumsum(pca.explained_variance_ratio_)
    #print(np.cumsum(pca.explained_variance_ratio_))
    
    #Interpretation of painting features
    plt.figure()
    plt.text(11,0.85,'%.2f'%cum_explained[8], ha='center',va='bottom', wrap=True)
    plt.plot([9, 10.5], [0.918,0.89], c='b', linestyle='-')
    plt.vlines(9,0,1, linestyle="dashed")
    index = np.arange(cum_explained.shape[0])+1
    plt.plot(index,cum_explained,marker = "o") 
    x = [2,4,5,6,9,10,12,14]
    plt.xticks(x)
    plt.ylim(0.3,None)
    plt.xlabel('# principal components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('./images/br_pca.png',dpi = 300)
    
    
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
        #print(r_value)
        
        ex_df.loc[i,'p_value'] = r_value
        ex_df_p.loc[i,'p_value'] = p_value
    
    

    
    plt.figure(figsize = (40,10))
    
    ax = sns.heatmap(ex_df, annot = ex_df_p, cmap="BuPu", annot_kws={"size":20},fmt = '')
    ax.yaxis.set_tick_params(labelsize=25,rotation = 0)
    ax.xaxis.set_tick_params(labelsize=25,rotation = 0)
    ### Set y-axis label        
    plt.savefig('./images/br_pca_heatmap.png', dpi=300)
    
    #set up dataframe index,columns
    BR_df = pd.DataFrame(ex, columns=BRmg_df.columns)
    L_sk = pd.DataFrame(L_sk, columns=BRmg_df.index)
    BR_df.index = [f"SILL{c}" for c in['一', '二','三','四','五','六','七','八','九']]
    L_sk.index = [f"BR{d}" for d in range(1,L_sk.shape[0]+1)]
    
    
    ex_df.to_csv('./outputs/br_pca.csv', index=True, encoding='big5')
    
    return(BR_df,L_sk,BRmg_df)

def normalization(df_ab):
    from sklearn import preprocessing
    nor = preprocessing.scale(df_ab)
    return(nor)

ex_br,BR_pca,BRmg_df = pca_br()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:40:17 2021

@author: peter
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import skimage.io as io
from plotnine import *
from mpl_toolkits import mplot3d
import matplotlib as mpl
from scipy import stats
from pandas.api.types import CategoricalDtype

file_path="./3D_face_data/"

feature_female=np.loadtxt(file_path+'Profile_AntFeature_female.txt') 

feature_male=np.loadtxt(file_path+'Profile_AntFeature_male.txt') 

#'face height/face width','jawline width/faced width','chin-to-month/jawline width'
df_female=pd.DataFrame(feature_female[:,0:4],  columns=['face height\nface width',
                                                        'jaw width\nfaced width',
                                                        'chin-to-mouth\njawline width',
                                                        'cheekbone width\nfaced width'])
df_female['group']='female'

df_male=pd.DataFrame(feature_male[:,0:4],  columns=['face height\nface width',
                                                        'jaw width\nfaced width',
                                                        'chin-to-mouth\njawline width',
                                                        'cheekbone width\nfaced width'])
df_male['group']='male'

df_feature=df_male.append(df_female)

df_melt=pd.melt(df_feature,id_vars='group')

df_melt['variable'] = df_melt['variable'].astype(CategoricalDtype(categories=[
                                  'chin-to-mouth\njawline width',
                                  'jaw width\nfaced width',
                                  'face height\nface width',
                                  'cheekbone width\nfaced width'],ordered=True))


colnames=df_female.columns[:-1]
ncol=len(colnames)
pstd=np.zeros((ncol))
pvalue=np.zeros((ncol))
for i in range(ncol):
    x=df_male.loc[:,colnames[i]]
    y=df_female.loc[:,colnames[i]]
    pstd[i]=stats.levene(x,y)[1]
    pvalue[i]=stats.ttest_ind(x,y,equal_var = False)[1]
    
df_pvalue=pd.DataFrame(dict(pstd=pstd,pvalue=pvalue,variable=colnames))

violin_plot=(ggplot()
 +geom_violin(aes(x='group',y="value",fill="group"),data=df_melt,show_legend=False)
 +geom_boxplot(aes(x='group',y="value",group="group"),data=df_melt,fill="white",width=0.1,show_legend=False)
 +scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='husl')
# +scale_fill_manual(values=("#F66EB8","#0199FC",))
 +facet_wrap('~variable',scales='free', ncol = 4)
 +ylab('Ratio')
 +xlab('Gender')
 +theme_matplotlib()
 +theme(strip_text_x=element_text(margin={'t': 20, 'r': 5},va='bottom'),
        strip_background=element_rect(color='k'),
        panel_spacing=0.5,
        figure_size=(10,3),
        dpi=300))

print(violin_plot)
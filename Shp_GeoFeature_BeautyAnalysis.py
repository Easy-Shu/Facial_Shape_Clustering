# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:24:41 2020

@author: -
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

#====================================Male:relationship between Geofeatures and beauty========================================

df_Mabeauty=pd.read_csv(file_path+"AM_Ratings_SCUT_FBP5500.csv")
df_Mabeauty=df_Mabeauty.groupby('filename',as_index=False)['Rating'].agg({'mean':"mean",'std':"std"})

params = []
with open(file_path+'Face_Para_male_SCUT_FBP5500.json', "r") as f:
    for line in f:
        doc = json.loads(line)
        #print(doc)
        params.append(doc) 
filenames=np.array([ x['filename'] for x in params])


labels=np.loadtxt(file_path+'AM_Keamlables.txt')
df_MaFeatures=pd.concat((pd.DataFrame(dict(filename=filenames,labels=labels)),df_male),axis=1)


df_MaMerge=pd.merge(df_MaFeatures,df_Mabeauty,how='left',on='filename')
df_MaMerge['labels']=df_MaMerge['labels'].astype(str)

# df_MaMerge['labels']=df_MaMerge['labels'].replace(['0.0', '1.0', '2.0', '3.0', '4.0', '5.0'],
#                                                    ['4' , '6' , '5', '2', '3', '1'])
#df_MaMerge.to_csv('./Data/df_MaMerge.csv',index=False)

violin_plot=(ggplot(df_MaMerge,aes(x='labels',y="mean",fill="labels"))
 +geom_violin(show_legend=False)
+geom_boxplot(fill="white",width=0.1,show_legend=False)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='husl')
+ylim(1,5)
+theme_matplotlib())

print(violin_plot)

#=====================================Female:relationship between Geofeatures and beauty========================================

df_febeauty1=pd.read_csv(file_path+"AF_Ratings_SCUT_FBP5500.csv")
df_febeauty1=df_febeauty1.groupby('filename',as_index=False)['Rating'].agg({'mean':"mean",'std':"std"})
df_febeauty2=pd.read_csv(file_path+'AF_Rating_SCUT_FBP.csv')
df_febeauty=df_febeauty1.append(df_febeauty2)


params = []
with open(file_path+'Face_Para_female_SCUT_FBP5500.json', "r") as f:
    for line in f:
        doc = json.loads(line)
        params.append(doc)
filenames1=np.array([ x['filename'] for x in params])

params = []
with open(file_path+'Face_Para_female_SCUT_FBP.json', "r") as f:
    for line in f:
        doc = json.loads(line)
        params.append(doc)
filenames2=np.array([ x['filename'] for x in params])
filenames=np.append(filenames1,filenames2)

labels=np.loadtxt(file_path+'AF_Keamlables.txt')
df_feFeatures=pd.concat((pd.DataFrame(dict(filename=filenames,labels=labels)),df_female),axis=1)


df_feMerge=pd.merge(df_feFeatures,df_febeauty,how='left',on='filename')
df_feMerge['labels']=df_feMerge['labels'].astype(str)
# df_feMerge['labels']=df_feMerge['labels'].replace(['0.0', '1.0', '2.0', '3.0', '4.0', '5.0'],
#                                                   ['2', '5', '1', '6', '4', '3'])
#df_feMerge.to_csv('./Data/df_feMerge.csv',index=False)

violin_plot=(ggplot(df_feMerge,aes(x='labels',y="mean",fill="labels"))
 +geom_violin(show_legend=False)
+geom_boxplot(fill="white",width=0.1,show_legend=False)
+scale_fill_hue(s = 0.90, l = 0.65, h=0.0417,color_space='husl')
+ylim(1,5)
+theme_matplotlib())

print(violin_plot)


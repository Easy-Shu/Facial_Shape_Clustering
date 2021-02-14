# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 21:49:54 2020

@author: 19110596R
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas  as pd
import seaborn as sns
from sklearn import manifold
import skimage.io as io
from plotnine import *
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans,AgglomerativeClustering, AffinityPropagation

colors_cluster=np.array(["#CC476B","#9F7000","#228B00","#009681","#0082CE","#B646C7","C7","C8","C9"])
 
gender="female"
df_profile=np.loadtxt('./Data/'+gender+'profile_lines.txt') 
labels=np.loadtxt('./Data/'+gender+"_Kmeans_labels.txt")
#female:(2493, 200)

#=====================================================================================
#====cluster num:3

tsne = manifold.TSNE(n_components=3, init='pca')
X_tsne = tsne.fit_transform(df_profile)

fig = plt.figure(figsize=(10,8),dpi =90)  
#ax =  fig.add_subplot(1, 1, 1,projection='3d')
ax = fig.gca(projection='3d')
#ax.set_aspect('equal','box')
ax.view_init(azim=75, elev=20)
##改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
for i in range(int(labels.max())+1):
    idx=np.where(labels==i)[0]
    ax.scatter3D(X_tsne[idx,0],X_tsne[idx,1],X_tsne[idx,2],
                c=colors_cluster[i],s=15,edgecolor='k',alpha=1)
plt.show()


#====cluster num:2
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(df_profile)

print("Org data dimension is {}. Embedded data dimension is {}".format(df_profile.shape[-1], X_tsne.shape[-1]))

# model = KMeans(n_clusters=6).fit(X_tsne)
# labels=model.labels_.astype(np.int8)
    

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

       
fig = plt.figure(figsize=(7,7), dpi=300)
for i in range(int(labels.max())+1):
    idx=np.where(labels==i)[0]
    plt.scatter(X_norm[idx, 0], X_norm[idx, 1],c=colors_cluster[i],marker='o',edgecolors='k',linewidths=0.5,alpha=1)
    plt.axis('off')
plt.show()

#===================================================================================
tri = Delaunay(X_tsne).simplices

fig = plt.figure(figsize=(15,15), dpi=300)
plt.triplot(X_norm[:,0], X_norm[:,1],tri,color='k',zorder=1,linewidth=1)
for i in range(int(labels.max())+1):
    idx=np.where(labels==i)[0]
    plt.scatter(X_norm[idx, 0], X_norm[idx, 1],c=colors_cluster[i],marker='o',edgecolors='k',linewidths=0.5,alpha=1,zorder=2)
    plt.axis('off')
plt.show()



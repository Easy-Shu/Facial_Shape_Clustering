# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:46:42 2020

@author: Peter_Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from sklearn.decomposition import PCA
from sklearn import preprocessing
import skfuzzy as fuzz
from sklearn.cluster import KMeans,AgglomerativeClustering
import matplotlib.pylab as pylab
from face3d.mesh.render import render_colors
from Shp_utils import add_light_BP,profile_line,Caculate_motion,profile_line_show,clustering_evaluate,plot_dendrogram

params = {'legend.fontsize': 13,
          'legend.title_fontsize': 14,
         'axes.labelsize': 16,
         'axes.titlesize':16,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
pylab.rcParams.update(params)

file_path="./3D_face_data/"
landmark_idx=np.array([21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914,
       48695, 49667, 50924, 52613, 33678, 33005, 32469, 32709, 38695,
       39392, 39782, 39987, 40154, 40893, 41059, 41267, 41661, 42367,
        8161,  8177,  8187,  8192,  6515,  7243,  8204,  9163,  9883,
        2215,  3886,  4920,  5828,  4801,  3640, 10455, 11353, 12383,
       14066, 12653, 11492,  5522,  6025,  7495,  8215,  8935, 10395,
       10795,  9555,  8836,  8236,  7636,  6915,  5909,  7384,  8223,
        9064, 10537,  8829,  8229,  7629, 
        32936, 42201, 41096, 40502, 39879, 38790, 21711], dtype=np.int64)
idx=landmark_idx[np.r_[range(17),71,57,69,73]]

h=500
w=500
max_size=110000

gender='female'
if gender=='male':
    colors0=np.load(file_path+'mat_colors_male.npy')
    vertices=np.load(file_path+'mean_vertices_Neutral_male.npy')
    
    
else:
    colors0=np.load(file_path+"mat_colors_female.npy")              #SCUT-FBP5500_v2+SCUT-FBP
    vertices=np.load(file_path+'mean_vertices_Neutral_female.npy')  #SCUT-FBP5500_v2+SCUT-FBP


mu_vertices=vertices.mean(1).reshape(-1,3)
mu_colors=colors0.mean(1).reshape(-1,3)
triangles=np.load(file_path+"BFM_triangles.npy")

#=================================profile landmarks selection=======================

X_ind_all = np.tile(idx[np.newaxis, :], [2, 1])*3
X_ind_all[1, :] += 1
valid_ladind = X_ind_all.flatten('F')

profile_vertices=vertices[valid_ladind,:]/1000

#=================================Normalize by distance between cheek points========================
# distance between left and right cheek points
ear_idx=np.array([1,15])
X_ind_all = np.tile(ear_idx[np.newaxis, :], [2, 1])*2
X_ind_all[1, :] += 1
valid_earind = X_ind_all.flatten('F')
ear_vertices=profile_vertices[valid_earind,:].T
ear_dist=np.array([ np.sqrt(np.sum((x[0:2]-x[2:])**2))  for x in ear_vertices])

#center between left and right cheek points
X_ind_all = np.tile(ear_idx[np.newaxis, :], [3, 1])*2
X_ind_all[1, :] += 1
X_ind_all[2, :] += 2
valid_earind = X_ind_all.flatten('F')
ear_vertices=profile_vertices[valid_earind,:].T    
ear_center=(ear_vertices[:,0:3]+ear_vertices[:,3:])/2      

vertices_t2=vertices-np.tile(ear_center.T,(int(len(vertices)/3),1))*1000
vertices_s=vertices_t2/ear_dist.reshape(1,-1)*ear_dist.mean()/1000


def caculate_ffeature(x):
    f1=np.linalg.norm(x[8,:]-x[17,:])/np.linalg.norm(x[0,:]-x[16,:])
    f2=np.linalg.norm(x[4,:]-x[12,:])/np.linalg.norm(x[0,:]-x[16,:])
    f3=np.linalg.norm(x[8,:]-x[18,:])/np.linalg.norm(x[4,:]-x[12,:])
    f4=np.linalg.norm(x[19,:]-x[20,:])/np.linalg.norm(x[0,:]-x[16,:])
    #f5_11=[np.abs((x[8,1]-x[i,1])/(x[8,0]-x[i,0])) for i in range(8)]
    f5_11=[ np.arctan(np.abs((x[8,1]-x[i,1])/(x[8,0]-x[i,0])))/np.pi*180 for i in range(8)]
    
    return np.r_[f1,f2,f3,f4,f5_11]

#=======================Anthropometry features calculation================================

profile_feature=np.array([caculate_ffeature(x.reshape(-1,2))  for x in profile_vertices.T]).reshape(-1,12)

#np.savetxt('./Data/'+gender+'_profile_feature_noscale.txt',profile_feature[:,0:4]) 

#===============Hierarchical clustering: structured vs unstructured ward==================
profile_feature=preprocessing.scale(profile_feature)
#np.savetxt('./Data/'+gender+'_profile_feature_scale.txt',profile_feature[:,0:4]) 
X=profile_feature[:,0:4]

#=========================================Clustering Result Visualization============================
distance_threshold=23#1.3#female:22  male: 20
# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=distance_threshold, 
                                n_clusters=None,linkage='ward')
model.fit(X)
labels=model.labels_.astype(np.int8)
n_clusters=model.n_clusters_

#model.labels_=model.labels_.astype(str)
#model.labels_[labels!=5]=" "
#np.savetxt('./Data/'+gender+'_labels.txt',labels)

fig = plt.figure(figsize=(15,5), dpi=300)
plt.title('Hierarchical Clustering Dendrogram',loc= 'left')
# plot the top three levels of the dendrogram
plot_dendrogram(model,color_threshold=distance_threshold)#, truncate_mode='level', p=5)
#plt.xlabel("Number of points in node (or index of point if no parenthesis).")
#plt.xticks(fontsize=0.5)
plt.show()
#==================================clustering result=====================

skin_type="texture"#"shape", "motion"  "texture"

for x in range(int(labels.max())+1):
    idx=np.where(labels==x)[0]
    print(len(idx))
    sub_colors=colors0[:,idx].mean(1).astype(float).reshape(-1,3)/255
    sub_vertices=vertices_s[:,idx].mean(1).reshape(-1,3)*1000
    
    sub_vertices=sub_vertices-sub_vertices[8192,:].reshape(1,3)
    
    
    dist_vertices=(sub_vertices-mu_vertices)/1000
    dist_vertices[:,0]=(np.abs(sub_vertices[:,0])-np.abs(mu_vertices[:,0]))/1000
    motion_color=Caculate_motion(dist_vertices,max_R=7)

    sub_vertices=sub_vertices-sub_vertices[8192,:].reshape(1,3)

    sub_vertices=sub_vertices*h/max_size/2
    sub_vertices[:,0]=sub_vertices[:,0]+w/2
    sub_vertices[:,1]=h/2-sub_vertices[:,1]
    #save_ply(sub_vertices,sub_colors,triangles,filePath4+'sub'+str(i)+'.ply')
    
    if skin_type=="texture":
        lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,
                            intensity_ambient=0.0,intensity_directional=0.15,
                            intensity_specular=0.15,light_pos=(-10,-10,100))# 
        
    elif skin_type=='motion':
        lig_colors=add_light_BP(sub_vertices, triangles,motion_color,
                               intensity_ambient=0.0,intensity_directional=0.3,
                               intensity_specular=0.1,light_pos=(-10,-10,100))# 
    
    elif skin_type=='shape' :
        sub_colors=np.repeat([100,100,100],len(mu_colors)).reshape(3,-1).T/255
        lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,
                                  intensity_ambient=0.0,intensity_directional=0.5,
                                  intensity_specular=0.3,light_pos=(-10,-10,100))
    
    
    image,_=render_colors(sub_vertices,triangles,lig_colors,h, w)
    
    image[image==0]=1.
    
    #io.imsave("./Data/"+gender+"_"+skin_type+str(x+1)+".png", (image*255).astype(np.uint8))
    
    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(image)#.astype(np.uint8))
    #plt.axis('off')
    #plt.xticks([])
    #plt.yticks([])
    #plt.savefig(filePath4+str(x)+"cluster.tiff")
    plt.show()
    
#np.savetxt(filePath4+'landmarks.txt',sub_vertices[landmark_idx,:])

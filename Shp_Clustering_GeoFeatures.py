# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:46:42 2020

@author: Peter_Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from sklearn.decomposition import PCA
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


#========================profile line calculation and reduce by PCA================================
profile_lines=np.array([ profile_line(x)[:-2]  for x in vertices_s.T]).reshape(-1,200)

mean_proflieline=profile_lines.mean(0).reshape(-1,2) 


profile_line_show(profile_lines,cluster_idx=None,color='r')


pca = PCA(n_components=2)
pca.fit(profile_lines)
#np.savetxt('./Data/'+gender+'profile_lines.txt',profile_lines) 

print(pca.explained_variance_ratio_,"sum:",np.sum(pca.explained_variance_ratio_))
print (pca.explained_variance_)

X2=pca.transform(profile_lines)


#============================================PCA Visualization==================

PC1_list=np.array([-45,-30,-15,0,15,30,45])

PC2_list=np.array([-20,-10,0,10,20])

PC_x=[]
PC_y=[]
PC_paths=[]
for i in range(len(PC1_list)):
    for j in range(len(PC2_list)):
        PC_x.append(i)
        PC_y.append(j)
        
        img_path="./Figures/"+gender+"_PCA_"+str(j+1)+"_"+str(i+1)+".png"
        PC_paths.append(img_path)
        
        x=PC1_list[i]
        y=PC2_list[j]
        
        idx=np.argsort((X2[:,0]-x)**2+(X2[:,1]-y)**2)[0:10]
            
        sub_colors=colors0.mean(1).astype(float).reshape(-1,3)/255
        sub_vertices=vertices_s[:,idx].mean(1).reshape(-1,3)*1000
        
        sub_vertices=sub_vertices-sub_vertices[8192,:].reshape(1,3)
         
        dist_vertices=(sub_vertices-mu_vertices)/1000
        dist_vertices[:,0]=(np.abs(sub_vertices[:,0])-np.abs(mu_vertices[:,0]))/1000
        motion_color=Caculate_motion(dist_vertices,max_R=7)
       
        sub_vertices=sub_vertices*h/max_size/2
        sub_vertices[:,0]=sub_vertices[:,0]+w/2
        sub_vertices[:,1]=h/2-sub_vertices[:,1]
        #save_ply(sub_vertices,sub_colors,triangles,filePath4+'sub'+str(i)+'.ply')
        
       
        lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,
                                intensity_ambient=0.0,intensity_directional=0.15,
                                intensity_specular=0.15,light_pos=(-10,-10,100))# 
            
        image,_=render_colors(sub_vertices,triangles,lig_colors,h, w)
        
        image[image==0]=1.
    
        
        fig = plt.figure(figsize=(6,6), dpi=300)
        plt.imshow(image)#.astype(np.uint8))
        plt.axis('off')
        
        #img_paths="./Data/"+gender+"_PCA_"+skin_type+str(j+1)+"_"+str(i+1)+".png"
        io.imsave(img_path, (image*255).astype(np.uint8))
    
        plt.show()
    
    
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def getImage(path,zoom=0.09):
    img = Image.open(path)
    img.thumbnail((512, 512), Image.ANTIALIAS) # resizes image in-place
    return OffsetImage(img,zoom=zoom)

 
fig, ax = plt.subplots(figsize=(5,3.5),dpi =300)
ax.scatter(PC_x, PC_y) 
plt.xlabel("X Axis",fontsize=0.1)
plt.ylabel("Y Axis",fontsize=0.1)
plt.xticks(fontsize=0.1)
plt.yticks(fontsize=0.1)
plt.gca().invert_yaxis()

artists = []
for x0, y0, path in zip(PC_x, PC_y,PC_paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    artists.append(ax.add_artist(ab))
 
#fig.savefig("./Data/"+gender+"_PC1_2.pdf")
plt.show() 

#np.savetxt('./Data/profile_vertices.txt',profile_feature[:,0:3]) 


#=========================================Clustering Result Evaluation============================
custering_method=[AgglomerativeClustering(),KMeans(),"FCM"]
k_max=6
#gap, reference_inertia, ondata_inertia = compute_gap(AgglomerativeClustering(), X, k_max)
#rho, delta=density_showing(X, dc=0.4)
score_SI=[]
score_CA=[]
for x in custering_method:
    num_clusters,score_SI0,score_CA0=clustering_evaluate(x, X2,X0=profile_lines, k_max=k_max)
    score_SI.append(score_SI0[-1])
    score_CA.append(score_CA0[-1])

score_SI=np.round(np.array(score_SI),2)    
score_CA=np.round(np.array(score_CA),2)  
#=========================================Clustering Result Visualization============================


method="Kmeans"

if method=="HC":
    #=======================Agglomerative Clustering========================
    distance_threshold=400#1.3#female:400  male: 300
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=distance_threshold, 
                                    n_clusters=None,linkage='ward')
    model.fit(X2)
    labels=model.labels_.astype(np.int8)
    n_clusters=model.n_clusters_
    
    threshold_min=model.distances_[model.distances_<distance_threshold].max()
    threshold_max=model.distances_[model.distances_>distance_threshold].min()
    print("threshold_min:",threshold_min, " threshold_max:",threshold_max)
    
    #np.savetxt('./Data/'+gender+"_"+method+'_labels.txt',labels)
 
    fig = plt.figure(figsize=(15,5), dpi=300)
    plt.title('Hierarchical Clustering Dendrogram',loc= 'left')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model,color_threshold=distance_threshold)#, truncate_mode='level', p=5)
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.xticks(fontsize=0.5)
    plt.show()
    
elif method=='Kmeans':
    #================================KMeans====================

    model = KMeans(n_clusters=6).fit(X2)
    labels=model.labels_.astype(np.int8)
    
    #np.savetxt('./Data/'+gender+"_"+method+'_labels.txt',labels)
    
elif method=="FCM":
    #=====================================FCM===================
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X2.T, 6, 2, error=0.0000001, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)

colors_cluster=np.array(["#CC476B","#9F7000","#228B00","#009681","#0082CE","#B646C7","C7","C8","C9"])

if method=="HC":
    idx_cluster=np.array(range(int(labels.max())+1))+1
    if gender=='male':
        colors_cluster=colors_cluster[np.array([2,4,3,6,1,5])-1]
    else:
        colors_cluster=colors_cluster[np.array([4,6,3,2,5,1])-1]
elif method=="Kmeans":
    if gender=='male':
        idx_cluster=np.array([4,6,5,2,3,1])
        colors_cluster=colors_cluster[idx_cluster-1]
    else:
        idx_cluster=np.array([2,5,1,6,4,3])
        colors_cluster=colors_cluster[idx_cluster-1]
        
fig = plt.figure(figsize=(7,7), dpi=300)
plt.axes([0.1,0.1,0.8,0.8])
for i in idx_cluster.argsort():
    idx=np.where(labels==i)[0]
    plt.scatter(X2[idx, 0], X2[idx, 1],c=colors_cluster[i],marker='o',edgecolors='k',linewidths=0.5,alpha=1,
                label='Cluster'+str(idx_cluster[i]))
plt.legend(title='Groups',loc='lower right',edgecolor='none',facecolor='w')
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.savefig('./Data/'+gender+"_"+method+'cluster_scatter.png', transparent=True)
#plt.show()


cluster_size=[]
for x in range(int(labels.max())+1):
    idx=np.where(labels==x)[0]
    cluster_size.append(len(idx))
cluster_size=np.array(cluster_size)[idx_cluster.argsort()] 

plt.axes([0.12,0.67,0.2,0.2])   
#fig = plt.figure(figsize=(7,7), dpi=300)
plt.pie(cluster_size,colors=colors_cluster[idx_cluster.argsort()] ,shadow=False, 
        startangle=90,counterclock =False,wedgeprops = {'linewidth': 0.5,'edgecolor':'k'})
plt.axis('equal') # 等价于 ax1.set(aspect='euqal')，使得饼图在figure窗口放大缩小的过程中，保持圆形不变。
plt.show()


# ###############################Clustering Result Visualization##############################################

skin_type="texture"#"shape", "motion"  "texture"

for x in range(int(labels.max())+1):
    idx=np.where(labels==x)[0]
    print(len(idx))
    sub_colors=colors0[:,idx].mean(1).astype(float).reshape(-1,3)/255

    sub_vertices=vertices_s[:,idx].mean(1).reshape(-1,3)*1000
    
    sub_vertices=sub_vertices-sub_vertices[8192,:].reshape(1,3)
     
    #write_ply(sub_vertices,(sub_colors*255).astype(np.uint8),triangles,"./Data/"+gender+"_"+method+"_"+str(x+1)+".ply")
    
    
    dist_vertices=(sub_vertices-mu_vertices)/1000
    dist_vertices[:,0]=(np.abs(sub_vertices[:,0])-np.abs(mu_vertices[:,0]))/1000
    motion_color=Caculate_motion(dist_vertices,max_R=7)

    
    sub_vertices=sub_vertices*h/max_size/2
    sub_vertices[:,0]=sub_vertices[:,0]+w/2
    sub_vertices[:,1]=h/2-sub_vertices[:,1]
    #
    
    if skin_type=="texture":
        lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,
                            intensity_ambient=0.0,intensity_directional=0.15,
                            intensity_specular=0.15,light_pos=(-10,-10,100))# 
        
    elif skin_type=='motion':
        lig_colors=add_light_BP(sub_vertices, triangles,motion_color,
                               intensity_ambient=0.0,intensity_directional=0.3,
                               intensity_specular=0.1,light_pos=(-10,-10,100))
    
    elif skin_type=='shape' :
        sub_colors=np.repeat([100,100,100],len(mu_colors)).reshape(3,-1).T/255
        lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,
                                  intensity_ambient=0.0,intensity_directional=0.5,
                                  intensity_specular=0.3,light_pos=(-10,-10,100))
    
    
    image,_=render_colors(sub_vertices,triangles,lig_colors,h, w)
    
    image[image==0]=1.
    
    sub_proflie=profile_line(sub_vertices.flatten()).reshape(-1,2) 
    
    
    
    profile_line_show(profile_lines,cluster_idx=idx,color=colors_cluster[x])
    
    
    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(image)#.astype(np.uint8))
    #plt.plot(sub_proflie[:,0],sub_proflie[:,1],zorder=2,c='r',alpha=1)
    plt.axis('off')
    #plt.xticks([])
    #plt.yticks([])
    #if skin_type=="shape":
    #    plt.savefig("./Data/"+gender+"_"+method+"_"+skin_type+str(x+1)+".png")
    #else:
    #io.imsave("./Data/"+gender+"_"+method+"_"+skin_type+str(x+1)+".png", (image*255).astype(np.uint8))
        
    plt.show()
    
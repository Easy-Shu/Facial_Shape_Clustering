# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:09:09 2020

@author: -
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:46:42 2020

@author: Peter_Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
#import skmisc #提供loess smoothing
from face3d.mesh.render import render_colors
from Shp_utils import add_light_BP,read_ply,profile_line,Caculate_motion,profile_line_show

    
landmark_idx=np.array([21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914,
            48695, 49667, 50924, 52613, 33678, 33005, 32469, 
            32709, 40502,8236]) #forehead landmarks

h=500
w=500
max_size=110000

file_path="./3D_face_data/"

#=========================================male mean face===============================================================
mu_verticesm,colorsm,triangles=read_ply(file_path+'mean_face_male.ply')
mu_verticesm=mu_verticesm-mu_verticesm[8192,:].reshape(1,3)

mu_verticesm1=mu_verticesm*h/max_size/2
mu_verticesm1[:,0]=mu_verticesm1[:,0]+w/2
mu_verticesm1[:,1]=h/2-mu_verticesm1[:,1]


lig_colors=add_light_BP(mu_verticesm1, triangles,colorsm.astype(float)/255,
                       intensity_ambient=0.0,intensity_directional=0.1,
                       intensity_specular=0.3,light_pos=(-10,-10,100))
 
imgtex_female,_=render_colors(mu_verticesm1,triangles,lig_colors,h, w)

#io.imsave("./Data/average_male_texture.png", (image*255).astype(np.uint8))

sub_colors=np.repeat([100,100,100],len(colorsm)).reshape(3,-1).T/255
lig_colors=add_light_BP(mu_verticesm1, triangles,sub_colors,
                          intensity_ambient=0.0,intensity_directional=0.5,
                          intensity_specular=0.3,light_pos=(-10,-10,100))

imgshp_female,_=render_colors(mu_verticesm1,triangles,lig_colors,h, w)
    
profile_female=profile_line(mu_verticesm1.flatten()).reshape(-1,2) 

# fig = plt.figure(figsize=(6,6), dpi=300)
# plt.imshow(imgshp_female)
# #plt.scatter(mu_vertices1[idx,0],mu_vertices1[idx,1],c='r',edgecolors='k',s=20)
# plt.plot(profile_female[:,0],profile_female[:,1],zorder=2,c='r',alpha=1)
# #plt.axis('off')
# #plt.xticks([])
# #plt.yticks([])
# #plt.savefig(filePath4+str(x)+"cluster.tiff")
# plt.show()

#===========================================female mean face============================================================
mu_verticesf,colorsf,triangles=read_ply(file_path+'mean_face_female.ply') # without expression
mu_verticesf=mu_verticesf-mu_verticesf[8192,:].reshape(1,3)

mu_verticesf1=mu_verticesf*h/max_size/2
mu_verticesf1[:,0]=mu_verticesf1[:,0]+w/2
mu_verticesf1[:,1]=h/2-mu_verticesf1[:,1]


lig_colors=add_light_BP(mu_verticesf1, triangles,colorsf.astype(float)/255,
                       intensity_ambient=0.0,intensity_directional=0.1,
                       intensity_specular=0.3,light_pos=(-10,-10,100))
imgtex_male,_=render_colors(mu_verticesf1,triangles,lig_colors,h, w)

#io.imsave("./Data/average_female_texture.png", (image*255).astype(np.uint8))

sub_colors=np.repeat([100,100,100],len(colorsf)).reshape(3,-1).T/255
lig_colors=add_light_BP(mu_verticesf1, triangles,sub_colors,
                          intensity_ambient=0.0,intensity_directional=0.5,
                          intensity_specular=0.3,light_pos=(-10,-10,100))

imgshp_male,_=render_colors(mu_verticesf1,triangles,lig_colors,h, w)
    
profile_male=profile_line(mu_verticesf1.flatten()).reshape(-1,2) 

# fig = plt.figure(figsize=(6,6), dpi=300)
# plt.imshow(imgshp_male)
# #plt.scatter(mu_vertices1[idx,0],mu_vertices1[idx,1],c='r',edgecolors='k',s=20)
# plt.plot(profile_male[:,0],profile_male[:,1],zorder=2,c='r',alpha=1)
# #plt.axis('off')
# #plt.xticks([])
# #plt.yticks([])
# #plt.savefig(filePath4+str(x)+"cluster.tiff")
# plt.show()

#=============================================average Asian face======================================================
mu_verticesa=(mu_verticesf+mu_verticesm)/2
colorsa=(colorsf.astype(float)+colorsm.astype(float))/2

mu_verticesa=mu_verticesa-mu_verticesa[8192,:].reshape(1,3)

mu_verticesa1=mu_verticesa*h/max_size/2
mu_verticesa1[:,0]=mu_verticesa1[:,0]+w/2
mu_verticesa1[:,1]=h/2-mu_verticesa1[:,1]


lig_colors=add_light_BP(mu_verticesa1, triangles,colorsa/255,
                       intensity_ambient=0.0,intensity_directional=0.1,
                       intensity_specular=0.3,light_pos=(-10,-10,100))
 
imgtex_global,_=render_colors(mu_verticesa1,triangles,lig_colors,h, w)
#io.imsave("./Data/average_all_texture.png", (image*255).astype(np.uint8))

sub_colors=np.repeat([100,100,100],len(colorsm)).reshape(3,-1).T/255
lig_colors=add_light_BP(mu_verticesa1, triangles,sub_colors,
                          intensity_ambient=0.0,intensity_directional=0.5,
                          intensity_specular=0.3,light_pos=(-10,-10,100))

imgshp_global,_=render_colors(mu_verticesa1,triangles,lig_colors,h, w)

proflie_global=profile_line(mu_verticesa1.flatten()).reshape(-1,2) 

# fig = plt.figure(figsize=(6,6), dpi=300)
# plt.imshow(imgshp_global)
# #plt.scatter(mu_vertices1[idx,0],mu_vertices1[idx,1],c='r',edgecolors='k',s=20)
# plt.plot(proflie_global[:,0],proflie_global[:,1],zorder=2,c='r',alpha=1)
# #plt.axis('off')
# #plt.xticks([])
# #plt.yticks([])
# #plt.savefig(filePath4+str(x)+"cluster.tiff")
# plt.show()

#====================================================motion image====================================================
dist_vertices=(mu_verticesm-mu_verticesa)/1000
dist_vertices[:,0]=(np.abs(mu_verticesm[:,0])-np.abs(mu_verticesa[:,0]))/1000
motion_color=Caculate_motion(dist_vertices,max_R=5)


lig_colors=add_light_BP(mu_verticesm1, triangles,motion_color,# colors.astype(float)/255,
                       intensity_ambient=0.0,intensity_directional=0.3,#male:0.3
                       intensity_specular=0.1,light_pos=(-10,-10,100))#male:0.5
    
    
imgmot_male,_=render_colors(mu_verticesm1,triangles,lig_colors,h, w)

#io.imsave("./Data/average_male_motion.png", (image*255).astype(np.uint8))

# fig = plt.figure(figsize=(6,6), dpi=300)
# plt.imshow(imgmot_male)#.astype(np.uint8))
# #plt.axis('off')
# #plt.xticks([])
# #plt.yticks([])
# #plt.savefig(filePath4+str(x)+"cluster.tiff")
# plt.show()

dist_vertices=(mu_verticesf-mu_verticesa)/1000
dist_vertices[:,0]=(np.abs(mu_verticesf[:,0])-np.abs(mu_verticesa[:,0]))/1000
motion_color=Caculate_motion(dist_vertices,max_R=5)


lig_colors=add_light_BP(mu_verticesf1, triangles,motion_color,# colors.astype(float)/255,
                       intensity_ambient=0.0,intensity_directional=0.3,#male:0.3
                       intensity_specular=0.1,light_pos=(-10,-10,100))#male:0.5
    
    
imgmot_female,_=render_colors(mu_verticesf1,triangles,lig_colors,h, w)

#io.imsave("./Data/average_female_motion.png", (image*255).astype(np.uint8))


#==================================================visualization=============================================
fig,ax = plt.subplots(3,3,figsize=(6,6), dpi=300)

ax[0,0].imshow(imgtex_global)
ax[0,1].imshow(imgtex_female)
ax[0,2].imshow(imgtex_male)
ax[1,0].imshow(imgshp_global)
ax[1,0].plot(proflie_global[:,0],proflie_global[:,1],zorder=2,c='r',alpha=1)
ax[1,1].imshow(imgshp_female)
ax[1,1].plot(profile_female[:,0],profile_female[:,1],zorder=2,c='r',alpha=1)
ax[1,2].imshow(imgshp_male)
ax[1,2].plot(profile_male[:,0],profile_male[:,1],zorder=2,c='r',alpha=1)
#ax[0,0].imshow(imgtex_global)
ax[2,1].imshow(imgmot_female)
ax[2,2].imshow(imgmot_male)
for i in range(3):
    for j in range(3):
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
plt.tight_layout(pad=0)
plt.show()


#=======================profile lines display=======================
profile_lines_female=np.load(file_path+'profile_lines_female.npy') 

profile_lines_male=np.load(file_path+'profile_lines_male.npy') 
profile_lines_all=np.r_[profile_lines_female,profile_lines_male]

profile_line_show(profile_lines_all,
                  cluster_idx=range(len(profile_lines_female)),
                  color='#F7746A')

profile_line_show(profile_lines_all,
                  cluster_idx=np.arange(len(profile_lines_male))+\
                  len(profile_lines_female),
                  color="#36ACAE")

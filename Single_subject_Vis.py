# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:59:33 2020

@author: -
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy.interpolate import splprep, splev
from face3d.mesh.render import render_colors
from Shp_utils import add_light_BP#,read_ply

file_path="./3D_face_data/"
landmark_line=np.array([21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914,
                       48695, 49667, 50924, 52613, 33678, 33005, 32469, 32709, 32681,
                       42517, 41965, 41468, 41097, 40818, 40528, 40241, 39967, 39570,
                       39113, 22470, 21844, 21873])

mat_vertices=np.load(file_path+"mean_vertices_female_SCUT_FBP.npy")
jsonname="Face_Para_female_SCUT_FBP.json"

df_info=[]
#k=-1
with open(file_path+jsonname, "r") as f:
    for line in f:
        #k=k+1
        doc = json.loads(line)
        df_info.append(doc)
        #if doc['filename']==filename:
            #param=doc
            #print(k,doc) 
            #break

k=7         
img_path="/media/peter/Data/Face_Image_Database/SCUT-FBP/Data_Collection/"          
#=====================================reconstructed face visualization with ground truth===================================================   
param=df_info[k]

filename=param['filename']  
sub_vertices=mat_vertices[:,k].reshape(-1,3)

s=param['s']
R=np.array(param['R']).reshape(3,3)
t=np.array(param['t']).reshape(1,3)
    
sub_vertices=sub_vertices.dot(R.T)*s+t
sub_vertices[:,1]=-sub_vertices[:,1]   

triangles=np.load(file_path+"BFM_triangles.npy")


img=io.imread(img_path+filename) 

img=img.astype(float)/255
h,w,c=img.shape

shp_colors=np.repeat([100,100,100],len(sub_vertices)).reshape(3,-1).T/255#motion_color#
lig_colors=add_light_BP(sub_vertices, triangles,shp_colors,
                        intensity_ambient=0.0,intensity_directional=0.5,#0.5
                        intensity_specular=0.3,light_pos=(-10,-10,100))# 

render_image,_=render_colors(sub_vertices,triangles,lig_colors,h, w)

img_back=img.copy()
img_back[render_image>0]=0
img_face=img.copy()
img_face[render_image==0]=0


w=0.7
render_image1=render_image*w+img_face*(1-w)+img_back

x=sub_vertices[landmark_line,0]
y=sub_vertices[landmark_line,1]
tck, u = splprep([x, y], s=0)
new_points0 = splev(np.arange(0,1.01,0.01), tck)

fig = plt.figure(figsize=(6,6), dpi=300)
plt.imshow(render_image1)#.astype(np.uint8))
plt.scatter(sub_vertices[landmark_line,0],sub_vertices[landmark_line,1],c='r',edgecolors='k',s=20,zorder=2)
plt.plot(new_points0[0],new_points0[1],zorder=1,c='r')
#plt.axis('off')
# plt.xticks([])
# plt.yticks([])
#plt.savefig(filePath4+str(x)+"cluster.tiff")
plt.show()

# render_image3=render_image.copy()
# render_image3[render_image==0]=img_back[render_image==0]
# fig = plt.figure(figsize=(6,6), dpi=300)
# plt.imshow(render_image3)#.astype(np.uint8))
# #plt.axis('off')
# #plt.xticks([])
# #plt.yticks([])
# #plt.savefig(filePath4+str(x)+"cluster.tiff")
# plt.show()

#=====================================reconstructed face visualization===================================================   
render_image2=render_image.copy() 
render_image2[render_image==0]=1.

#io.imsave("./Data/mcluster0-0"+str(x)+".png", (image*255).astype(np.uint8))
fig = plt.figure(figsize=(6,6), dpi=300)
plt.imshow(render_image2)#.astype(np.uint8))
#plt.axis('off')
plt.xticks([])
plt.yticks([])
#plt.savefig(filePath4+str(x)+"cluster.tiff")
plt.show()

#==========================Corrected reconstructed face visualization with facila profile====================================================
s=param['s']
R=np.array(param['R']).reshape(3,3)
t=np.array(param['t'])

sub_vertices[:,1]=-sub_vertices[:,1]   
sub_vertices=((sub_vertices-t.reshape(1,3))/s).dot(np.linalg.inv(R.T))
    
h=500
w=500 

max_size=110000#(np.abs(sub_vertices[:,0:2])).max()
sub_vertices=sub_vertices-sub_vertices[8192,:].reshape(1,3)
    

sub_vertices=sub_vertices*h/max_size/2
sub_vertices[:,0]=sub_vertices[:,0]+w/2
sub_vertices[:,1]=h/2-sub_vertices[:,1]
#save_ply(sub_vertices,sub_colors,triangles,filePath4+'sub'+str(i)+'.ply')

#sub_colors=sucolors1.astype(float)/255#
sub_colors=np.repeat([100,100,100],len(sub_vertices)).reshape(3,-1).T/255#motion_color#
lig_colors=add_light_BP(sub_vertices, triangles,sub_colors,# colors.astype(float)/255,
                        intensity_ambient=0.0,intensity_directional=0.5,#male:0.3
                        intensity_specular=0.3,light_pos=(-10,-10,100))#male:0.5

image,_=render_colors(sub_vertices,triangles,lig_colors,h, w)

image[image==0]=1.

#io.imsave("./Data/mcluster0-0"+str(x)+".png", (image*255).astype(np.uint8))

x=sub_vertices[landmark_line,0]
y=sub_vertices[landmark_line,1]

tck, u = splprep([x, y], s=0)
new_points1 = splev(np.arange(0,1.01,0.01), tck)

fig = plt.figure(figsize=(6,6), dpi=300)
plt.imshow(image)#.astype(np.uint8))
#plt.plot(sub_vertices[landmark_line,0],sub_vertices[landmark_line,1],c='k',zorder=1)
plt.scatter(sub_vertices[landmark_line,0],sub_vertices[landmark_line,1],c='r',edgecolors='k',s=20,zorder=2)
plt.plot(new_points1[0],new_points1[1],zorder=1,c='r')
#plt.axis('off')
plt.xticks([])
plt.yticks([])
#plt.savefig(filePath4+str(x)+"cluster.tiff")
plt.show()    
    
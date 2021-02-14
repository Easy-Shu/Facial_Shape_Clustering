#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:46:36 2021

@author: peter
"""
import numpy as np
import colorsys
import pandas  as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.api.types import CategoricalDtype
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from collections import defaultdict


_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]

def get_normal(vertices, triangles):
    ''' calculate normal direction in each vertex
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    Returns:
        normal: [nver, 3]
    '''
    pt0 = vertices[triangles[:, 0], :] # [ntri, 3]
    pt1 = vertices[triangles[:, 1], :] # [ntri, 3]
    pt2 = vertices[triangles[:, 2], :] # [ntri, 3]
    #tri_normal = np.cross(pt0 - pt1, pt0 - pt2) # [ntri, 3]. normal of each triangle
    tri_normal = np.cross(pt1 - pt0, pt2 - pt0) # [ntri, 3]. normal of each triangle
    d = np.sqrt( np.sum(tri_normal *tri_normal, 1))
    zero_ind = (d == 0)
    d[zero_ind] = 1
    tri_normal=tri_normal/d.reshape(-1,1)
    
    normal = np.zeros_like(vertices) # [nver, 3]
    for i in range(triangles.shape[0]):
        normal[triangles[i, 0], :] = normal[triangles[i, 0], :] + tri_normal[i, :]
        normal[triangles[i, 1], :] = normal[triangles[i, 1], :] + tri_normal[i, :]
        normal[triangles[i, 2], :] = normal[triangles[i, 2], :] + tri_normal[i, :]
    
    # normalize to unit length
    mag = np.sum(normal**2, 1) # [nver]
    zero_ind = (mag == 0)
    mag[zero_ind] = 1;
    normal[zero_ind, 0] = np.ones((np.sum(zero_ind)))

    normal = normal/np.sqrt(mag[:,np.newaxis])

    return normal

def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj

def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices

# Reference: https://github.com/cleardusk/3DDFA/blob/master/utils/lighting.py
def add_light_BP(vertices, triangles, colors, **kwargs):
    #BlinnPhong光照模型
    intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.3))
    intensity_directional = convert_type(kwargs.get('intensity_directional', 0.6))
    intensity_specular = convert_type(kwargs.get('intensity_specular', 0.9))
    specular_exp = kwargs.get('specular_exp', 5)
    color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
    color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
    light_pos = convert_type(kwargs.get('light_pos', (0, 0, 1)))
    view_pos = convert_type(kwargs.get('view_pos', (0, 0, 1)))
    
    assert vertices.shape[0] == colors.shape[0]
    normal = get_normal(vertices, triangles) # [nver, 3]
    
    lit_colors=colors.copy()
    # ambient component 环境光
    #if intensity_ambient > 0:
    lit_colors += intensity_ambient * color_ambient
        
    vertices_n = norm_vertices(vertices.copy())  
    
    if intensity_directional > 0:
    # diffuse component 漫反射
        direction = _norm(light_pos - vertices_n)
        cos = np.sum(normal * direction, axis=1)[:, None]
        # cos = np.clip(cos, 0, 1)
        #  todo: check below
        lit_colors += intensity_directional * (color_directional * np.clip(cos, 0, 1))

    # specular component 镜面反射
    if intensity_specular > 0:
        v2v = _norm(view_pos - vertices_n)
        reflection = 2 * cos * normal - direction
        spe = np.sum((v2v * reflection) ** specular_exp, axis=1)[:, None]
        spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
        lit_colors += intensity_specular * color_directional * np.clip(spe, 0, 1)
        
        
    lit_colors = np.clip(lit_colors, 0, 1)
        
    return lit_colors


def Caculate_motion(dist_vertices,max_R=5):
    R=np.sqrt(dist_vertices[:,0]**2+dist_vertices[:,1]**2)
    angle=np.arctan(dist_vertices[:,1]/dist_vertices[:,0])
    
    angle_direction=np.array([ np.pi if x<0 else 0 for x in dist_vertices[:,0]]) 
    angle=angle+angle_direction
    
    Vs=dist_vertices[:,2]/max_R
    Vs[Vs<0.5]=0.5
    Vs[Vs>0.9]=0.9
    motion_color=np.array([colorsys.hsv_to_rgb(h, s, v) for h,s,v in zip(angle/(2*np.pi),R/max_R,Vs)])
    
    #motion_color=np.array([colorsys.hsv_to_rgb(h, s, 0.6) for h,s in zip(angle/(2*np.pi),R/max_R)])
    return motion_color


def Motion_colorCircle(Vs=0.6,Ss_max=10,method="plotnine"):
    #Color Motion Legend: Motion_colorCircle()
    #Vs=0.6#np.repeat(0.6,len(Rs))
    Hs=np.arange(0,360,1)
    Ss=np.arange(0,Ss_max,0.1)/Ss_max
    #Ss=Ss/Ss.max()
    
    RGBs=[]
    x=[]
    y=[]
    for h in Hs:
        for s in Ss:
            rgb=(np.array(colorsys.hsv_to_rgb(h/360,s,Vs))*255).astype(np.uint8)
            RGBs.append( "#{:02x}{:02x}{:02x}".format(rgb[0],rgb[1],rgb[2]))
            x.append(s*np.cos(h*np.pi/180))
            y.append(s*np.sin(h*np.pi/180))   
    df_colors=pd.DataFrame(dict(x=x,y=y,RGB=RGBs))
    
    if method=="plotnine":
        from plotnine import ggplot,aes,geom_point,scale_fill_manual,coord_fixed
        df_colors['order'] = range(len(df_colors))
        df_colors['order'] = df_colors['order'].astype(CategoricalDtype(categories=df_colors['order'],ordered=True))
        
        base_plot=(ggplot(df_colors,aes(x='x',y='y',fill='order'))+
                    geom_point(shape='o',color='none',show_legend=False)+
                    scale_fill_manual(values=df_colors['RGB'])+
                    coord_fixed(ratio = 1)
                    )
        print(base_plot)
    else:
        fig = plt.figure(figsize=(6,6), dpi=300)
        plt.scatter(x,y,s=3,c=RGBs,marker='o')
        plt.axis('square')
        plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    labels=model.labels_.astype(np.int8)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

class Clusters(dict):
    from matplotlib.colors import rgb2hex, colorConverter
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html
    
def get_cluster_classes(den, label='ivl'):
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

def profile_line_show(profile_lines,cluster_idx=None,color='r'):
    fig = plt.figure(figsize=(6,6), dpi=300)
    
    ax = plt.gca()                                            # get current axis 获得坐标轴对象
    
    ax.spines['right'].set_color('none') 
    ax.spines['top'].set_color('none')         # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    
    ax.xaxis.set_ticks_position('bottom')   
    ax.yaxis.set_ticks_position('left')          # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    
    ax.spines['bottom'].set_position(('data', 0))   #指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    ax.set_aspect('equal')
    for i in range(len(profile_lines)):
        new_points=profile_lines[i,:].reshape(-1,2)
        new_points=np.r_[new_points,new_points[0:1,:]]
        plt.plot(new_points[:,0],new_points[:,1],zorder=1,c='gray',alpha=1,linewidth=1)
    
        
    if cluster_idx is None:
        mean_proflieline=profile_lines.mean(0).reshape(-1,2) 
        mean_line=np.r_[mean_proflieline,mean_proflieline[0:1,:]]
        plt.plot(mean_line[:,0],mean_line[:,1],zorder=2,c='k',alpha=1)
    else:
        for i in cluster_idx:
            new_points=profile_lines[i,:].reshape(-1,2)
            new_points=np.r_[new_points,new_points[0:1,:]]
            plt.plot(new_points[:,0],new_points[:,1],zorder=1,c=color,alpha=1)
            
        mean_proflieline=profile_lines[cluster_idx,:].mean(0).reshape(-1,2) 
        mean_line=np.r_[mean_proflieline,mean_proflieline[0:1,:]]
        plt.plot(mean_line[:,0],mean_line[:,1],zorder=2,c='k',alpha=1)
    plt.show()  
        
    
def profile_line(sub_vertices):
    
    landmark_line=np.array([21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914,
       48695, 49667, 50924, 52613, 33678, 33005, 32469, 32709, 32681,
       42517, 41965, 41468, 41097, 40818, 40528, 40241, 39967, 39570,
       39113, 22470, 21844, 21873])
    
    x=sub_vertices[landmark_line*3]
    y=sub_vertices[landmark_line*3+1]
    
    tck, u = splprep([x, y], s=0)
    new_points = splev(np.arange(0,1.01,0.01), tck)
    vect_prof=(np.c_[new_points[0],new_points[1]]).reshape(-1,1)
    return vect_prof

def read_ply(filename):
    from plyfile import PlyData
    
    plydata = PlyData.read(filename)

    tri_data = plydata['face'].data['vertex_indices']
    triangles = np.vstack(tri_data)  # MX3
    vertex_color = plydata['vertex'] 
    (x, y, z,r,g,b) = (vertex_color[t] for t in ('x', 'y', 'z','red', 'green', 'blue'))

    colors=np.array([r,g,b]).T
    vertices=np.array([x,y,z]).T  # NX3
    
    #if flag_show:
    #    plot_mlabvertex(vertices,colors,triangles)
    return vertices,colors,triangles


def clustering_evaluate(model, X,X0=None, k_max=5, flag_show=True):
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    import skfuzzy as fuzz
    #from fcmeans import FCM
    
    if X0 is None:
        X0=X
        
    score_SI=[]
    score_CA=[]
    #distortions = []
    #Inertia=[]
    #Max_num=16
    for k in range(2,k_max+1):
        
        #model = AgglomerativeClustering(n_clusters=num_cluster,linkage='ward').fit(X)
        if model=="FCM":
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X.T, k, 2, error=0.00000001, maxiter=1000, init=None)
            
            labels = np.argmax(u, axis=0)
        else:
            model.n_clusters = k
            model.fit(X)
            labels=model.labels_
        
        score_SI0=silhouette_score(X0,labels)
        score_SI.append(score_SI0)
        #print('聚类%d簇的silhouette_score%f'%(num_cluster,score_SI0))
    
        score_CA0=calinski_harabasz_score(X0,labels)
        score_CA.append(score_CA0)
        print('聚类%d簇的calinski_harabaz: %f,silhouette_score: %f'%(k,score_CA0,score_SI0))
        
        #https://medium.com/@masarudheena/4-best-ways-to-find-optimal-number-of-clusters-for-clustering-with-python-code-706199fa957c
        #distortions.append(sum(np.min(cdist(X, model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        #Inertia.append(model.inertia_)
        
    fig = plt.figure(figsize=(7,5), dpi=300)
    plt.plot(range(2,k_max+1),score_SI,zorder=1,c='k')
    plt.scatter(range(2,k_max+1),score_SI,c='r',edgecolors='k',s=20,zorder=2)
    plt.show()
    
    fig = plt.figure(figsize=(7,5), dpi=300)
    plt.plot(range(2,k_max+1),score_CA,zorder=1,c='k')
    plt.scatter(range(2,k_max+1),score_CA,c='r',edgecolors='k',s=20,zorder=2)
    plt.show()
    
    return range(2,k_max+1),score_SI,score_CA



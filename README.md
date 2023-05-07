# [3D-Guided Facial Shape Clustering and Analysis](https://www.researchgate.net/publication/358382848)
## Introduction
Facial shape classification is of crucial importance in facial characteristics analysis and product recommendation. In this paper, we develop a 3D-guided facial shape clustering and analysis method to classify facial shapes without supervision, which is more reliable and accurate. This method consists of four steps: 3D face reconstruction, facial shape normalization, facial feature extraction and facial contour clustering. 

## 2D Image Databases
In our facial shape analysis, 1997 Asian male and 1993 Asian female face images are selected from [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release) and 500 Asian female images are selected from  [SCUT-FBP](http://www.hcii-lab.net/data/SCUT-FBP/), because there are several 2D image whose face cannot be detected successfully for their lighting problems.  Hence, 1997 Asian male and 2443 Asian female face images are used for further 3D face reconstruction.

<! ## Principle Component Analysis
<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure_6_PCA_Female.png" width=80% alt="Female"></img>
<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure_6_PCA_Male.png" width=80% alt="Male"></img>
>

## Shape Clustering Results
The Circular Dendrogram of the hierarchical clustering result based on geometric features shows the optimal cluster number is 6 for 3D female and male faces and the experimental results demonstrate the K-means clustering on geometric features can achieve the better performance. The average facial shapes of clusters can be downloaded from [shape](https://github.com/Easy-Shu/facial_shape_clustering/tree/main/Shapes) file. **Note: the cluster order of the K-means clustering results are different in each time.**

<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure13 Female_Clsutering_Results.png" width=100% alt="beasuty"></img>
Figure 13. Average 3D female facial shape of different clusters by K-means clustering method. The first and second row:  Average facial shape with texture and without texture. The third row: motion mesh from source Tf to target Ti (Ti-Tf), where i =1,2,..,6. The forth row: colour contours of cluster with the black average contour in the all grey facial contours.


<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure14 Male_Clsutering_Results.png" width=100% alt="beasuty"></img>
Figure 14. Average 3D male facial shape of different clusters by K-means clustering method. The first and second row:  Average facial shape with texture and without texture. The third row: motion mesh from source Ta to target Ti (Ti-Ta), where i =1,2,..,6. The forth row: colour contours of cluster with the black average contour in the all grey facial contours.

## Cluster Beauty Analysis

Aattractiveness distributions do not accord with normal distribution in the violin plot of the below Figure. So, the Wilcoxon signed-rank tests are used to compared the difference between two neighbor clusters. The investigation of the attractiveness distribution of different facial shape clusters reveals that the facial shapes with more pointed chin have higher attractiveness, regardless of male and female.  

<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure13_Beauty_analysis.png" width=100% alt="beasuty"></img>

## Acknowledgement
* Previous works on 3D dense face alignment or reconstruction: [PRNet](https://github.com/YadiraF/PRNet), [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2), [facescape](https://github.com/zhuhao-nju/facescape),  [face3d](https://github.com/YadiraF/face3d).
* Circular Dendrogram realization: [circlize: circular visualization in R](https://github.com/jokergoo/circlize). 

## Citation
If you find our work useful to your research, please consider citing:
```
@article{Jie2022clustering,
  title={{3D}-guided facial shape clustering and analysis},
  author={Zhang, Jie and Zhou, Kangneng and Luximon, Yan and Li, Ping and Iftikhar, Hassan},
  journal={Multimedia Tools and Applications},
  volume={81},
  number={6},
  pages={8785--8806},
  year={2022},
  publisher={Springer}
}
```
## Contacts
Please contact  jpeter.zhang@connect.polyu.hk  or open an issue for any questions or suggestions.

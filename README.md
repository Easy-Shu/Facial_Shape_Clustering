# 3D-guided facial shape clustering and analysis
## Introduction
Facial shape classification is of crucial importance in facial characteristics analysis and product recommendation. In this paper, we develop a 3D-guided facial shape clustering and analysis method to classify facial shapes without supervision, which is more reliable and accurate. This method consists of four steps: 3D face reconstruction, facial shape normalization, facial feature extraction and facial contour clustering. Firstly, we incorporate two 3D face reconstruction methods to reconstruct 3D face mesh without expression component from 1997 male and 2493 female facial images. Secondly, we normalize these 3D facial contours by translation and scaling. Thirdly, we propose two facial contour representations: geometric and anthropometric features. Fourthly, we use and compare three clustering methods to cluster these facial contours based on the extracted contour features by using Silhouette Coefficient and Calinski-Harabasz Index. The Circular Dendrogram of the hierarchical clustering result based on geometric features shows the optimal cluster number is 6 for 3D female and male faces and the analysis results demonstrate the K-means clustering on geometric features can achieve better performance. A further investigation between the beauty distribution and facial shape clusters reveals that the facial shapes with more pointed chin have higher beauty ratings, regardless of male or female. The facial shape analysis results can be applied in face-related product design, hairstyle recommendation and cartoon character design. The code will be released to the public for research purpose. 

## Principle Component Analysis

<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure_6_PCA_Female.png" width=80% alt="Female"></img>
<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure_6_PCA_Male.png" width=80% alt="Male"></img>

## Shape Clustering Results
The Circular Dendrogram of the hierarchical clustering result based on geometric features shows the optimal cluster number is 6 for 3D female and male faces and the experimental results demonstrate the K-means clustering on geometric features can achieve the better performance. The average facial shapes of clusters can be downloaded from [shape](https://github.com/Easy-Shu/facial_shape_clustering/tree/main/Shapes) file. **Note: the cluster order of the K-means clustering results are different in each time.**


## Cluster Beauty Analysis

Aattractiveness distributions do not accord with normal distribution in the violin plot of the below Figure. So, the Wilcoxon signed-rank tests are used to compared the difference between two neighbor clusters. The investigation of the attractiveness distribution of different facial shape clusters reveals that the facial shapes with more pointed chin have higher attractiveness, regardless of male and female.  

<img src="https://github.com/Easy-Shu/facial_shape_clustering/blob/main/Figures/Figure13_Beauty_analysis.png" width=100% alt="beasuty"></img>

## Acknowledgement
* Previous works on 3D dense face alignment or reconstruction: [PRNet](https://github.com/YadiraF/PRNet), [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2), [facescape](https://github.com/zhuhao-nju/facescape),  [face3d](https://github.com/YadiraF/face3d).
* Circular Dendrogram realization: [circlize: circular visualization in R](https://github.com/jokergoo/circlize). 

## Contacts
Please contact  jpeter.zhang@connect.polyu.hk  or open an issue for any questions or suggestions.

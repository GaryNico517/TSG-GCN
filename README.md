# Individual Graph Representation Learning for Pediatric Tooth Segmentation from Dental CBCT
_By Yusheng Liu<sup>a</sup>(lys_sjtu@sjtu.edu.cn), Shu Zhang<sup>b</sup>, Xiyi Wu<sup>a</sup>, Tao Yang<sup>a</sup>, Yuchen Pei<sup>a</sup>, Huayan Guo<sup>e</sup>, Yuxian Jiang<sup>a</sup>, Zhien Feng<sup>c</sup>, Wen Xiao<sup>d</sup>, Yu-Ping Wang<sup>e</sup>, and Lisheng Wang<sup>a</sup>(lswang@sjtu.edu.cn)_

_a.Department of Automation, Shanghai Jiao Tong University, Shanghai 200240, People’s Republic of China._  
_b.Department of Pediatric Dentistry, Beijing Stomatological Hospital, Capital Medical University, Beijing 100050, China._  
_c.Department of Oral and Maxillofacial-Head and Neck Oncology, Beijing Stomatological Hospital, Capital Medical University, Beijing 100050, China._  
_d.Department of Pediatric Dentistry, Shanghai Ninth People’s Hospital, Shanghai 200011, China._    
_e.Department of Dentistry, Shanghai East Hospital Affiliated to Tongji University, Shanghai 200120, China._  
_f.The Biomedical Engineering Department, Tulane University, New Orleans, LA 70118 USA._  

## Introduction
**This code repository is for our paper 'Individual Graph Representation Learning for Pediatric Tooth Segmentation from Dental CBCT' published by IEEE Transactions on Medical Imaging.**


**The pipeline of the proposed segmentation framework is shown in Figure.A below.**     
Our method will first segment four quadrants from a CBCT by a trained UNet. Subsequently, TSG-GCN segments teeth within each quadrant. The aggregation of teeth from the four quadrants yields the final segmentation results.   
![](Pipeline.PNG)

**In this section, we present the TSG-GCN architecture characterized by the encoder-decoder framework.**    
It incorporates a unified encoder, a 2D projection decoder for adaptive adjacency matrix learning, and a 3D GCN-driven decoder tailored fo nuanced multi-class teeth segmentation, as depicted in Figure.B below.       
![](Framework.PNG)

## Environments Configuration & Data Preparation
This repository is based on PyTorch 1.12.1.  
1. Clone the repository, and then install the related dependencies with pip.
```
git clone https://github.com/GaryNico517/TSG-GCN.git
cd TSG-GCN
pip install -r requirements.txt
```
2. Preparing your own dataset by segmenting four quadrants from CBCTs using your customized code, and relabeling the teeth as background(0), permanent(1-8), deciduous(9-13), supernumerary teeth(14) and irrelevant teeth(15).
## Training Procedure
### Training Quadrant Segmentation Network
1. Relabel the teeth in four quadrant of the full-scale CBCTs as 1, 2, 3, 4, respectively.
2. Train the Quadrant Segmentation Network using whatever model you prefer, including 3DUNet, nnUNet, and so forth.
### Training Topology Structure-guided Graph Convolutional Network (TSG-GCN)
1. 

## Inference Procedure

## Supplemental Weight:
### nnUNet-based
The training weights are for 1) Quadrant-wise Localization as well as 2) Tooth Segmentation on Quadrant-wise Data.  
The dataset used for training is a publicly available teeth dataset, containing:  
a) 76 CBCT images come from https://github.com/ErdanC/Tooth-and-alveolar-bone-segmentation-from-CBCT/  
b) 63 CBCT images come from https://ditto.ing.unimore.it/toothfairy2/  
You can easliy download the weights from https://drive.google.com/drive/folders/1-Kkm0C5huUZu_T04ciR5hRTn7G6lZbpd?usp=drive_link

## Citations (Updating)
If you find this repository useful, please consider citing our paper:  
```
 DOI: 10.1109/TMI.2024.3501365
```



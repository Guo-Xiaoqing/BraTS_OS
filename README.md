# BraTS_OS
Feature extraction for overall survival prediction in BraTS 2019 challenge
## Requirements
* Nibabel
* Openpyxl
* opencv3

## Feature explanation
We define 36 hand-crafted features that involves non-image features and image features.   
a) Non-image features includes age and resection status.  
b) Image featues includes vlolume, volume ratio, surface area，surface area to volume ratio，position of the whole tumor  center，position of the enhancing tumor center, relevant location of the whole tumor center to brain centerand relevant location of theenhancing tumor center to brain center.  
  
  
For more details of feature defination, you can refer to our paper: Domain Knowledge Based Brain Tumor Segmentation and Overall Survival Prediction

## How-to
1. Run ``` python feature_extraction.py ```, and you can generate your own feature excel

## Results
we train a GBDT model on hand-crafted features and position encoding segmentation. The validation and test results of Brats 2019 survival prediction are shown below:  
![image](https://github.com/Guo-Xiaoqing/BraTS_OS/blob/master/feature_excel/Brats_valid.png)  
![image](https://github.com/Guo-Xiaoqing/BraTS_OS/blob/master/feature_excel/Brats_test.png)  



# BraTS_OS
Feature extraction for overall survival prediction in BraTS 2019 challenge
## Requirements
* Nibabel
* Openpyxl
* opencv3

## Feature explanation
We define 36 hand-crafted features that involves non-image features and image features.   
a) Non-image features includes age and resection status.  
b) Image featues includes vlolume, volume ratio, surface area，surface area to volume ratio，position of the whole tumor  center，position of the enhancing tumor center, relevant location of the whole tumor center to brain center (3 coordinates) and relevant location of theenhancing tumor center to brain center (3 coordinates).  
More details of feature defination, you can refer to our paper: Domain Knowledge Based Brain TumorSegmentation and Overall Survival Prediction

## How-to
1. Run ``` python feature_extraction.py ```, and you can generate your own excel

## 

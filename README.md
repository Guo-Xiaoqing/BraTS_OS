# BraTS_OS
Feature extraction for overall survival prediction in BraTS 2019 challenge
## Requirements
* Nibabel
* Openpyxl
* opencv3

## Feature explanation
We define 36 hand-crafted features that involves non-image features and image features.   
a) Non-image features includes age and resection status.  
b) Image featues includes vlolume, volume ratio, 
![](http://latex.codecogs.com/gif.latex?\\\frac{V_{whole}}{V_{brain}})

## How-to
1. Run ``` python feature_extraction.py ```, and you can generate your own excel

## 

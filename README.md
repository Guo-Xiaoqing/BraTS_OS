# BraTS_OS
## Summary:
* This repository is for ["Domain Knowledge Based Brain Tumor Segmentation and Overall Survival Prediction"](https://link.springer.com/chapter/10.1007/978-3-030-46643-5_28). 
* Feature extraction for overall survival prediction in BraTS 2019 challenge.

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

## Citation:
If you found this repository helpful for your research, please cite our paper:
```
@inproceedings{guo2019domain,
  title={Domain Knowledge Based Brain Tumor Segmentation and Overall Survival Prediction},
  author={Guo, Xiaoqing and Yang, Chen and Lam, Pak Lun and Woo, Y.M. Peter and Yuan, Yixuan},
  booktitle={BrainLes},
  volume={11993},
  year={2019},
  organization={Springer}
}
```

## Questions:
Please contact "xiaoqingguo1128@gmail.com" or "cyang.ee@my.cityu.edu.hk"

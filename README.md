# AI-Powered Identification of High-Risk Breast Cancer Subgroups Using Mammography: A Multicenter Study Integrating Automated Brightest Density Measures with Deep Learning Metrics

## Create time : 2023. 09. 15

## Introduction
- This repository contains the source code for developing a deep-learning based method for multi-level MBD measurement (MIDAS) and risk prediction.
- Codes are for
  - Image preprocessing 
  - Module 1 : MIDAS (semantic segmentation of multi-level MBD)
  - Module 2 : Deep learning model for binary classification (estimating feature score (FS))
  - Statistical analysis
  - XAI (Guided Grad-CAM) visualization 

## 1) Image Preprocessing 
- To preprocess raw DICOM files, functions in `utils/image_preprocessing.py` are used.
  - For images used for Module 1, use `preprocess_dicom_seg` function.
    - Input : DICOM, 3-level thresholds for multi-level MBD
    - Output : Preprocessed image (in numpy array format), 3-level Multi-level MBD ground truth (in numpy array format)
  - For images used for Module 2, use `preprocess_dicom` function
    - Input : DICOM
    - Output : Preprocessed image (in numpy array format)
  - [PyDicom](https://pydicom.github.io/) library is needed for code execution. 

## 2) Module 1 : MIDAS (Automated multi-level MBD estimation) 
- Using preprocessed image and corresponding 3-level Multi-level MBD ground truth mask (binary), we used [improved attention u-net](https://github.com/nabsabraham/focal-tversky-unet)
- Model was implemented with Keras API (tensorflow v 2.5) 
- Requirements are listed in `requirements/module1.txt`
  - Installation : `pip install -r module1.txt`
- Example code for training MIDAS in `scripts/train_MIDAS.py`
- You can download trained segmentation networks (used in this study) in below link
  - [Cumulus segmentation network](https://drive.google.com/file/d/1HTpdhpmr0Hy9NWSLysOs0PaXNqBqRIRK/view?usp=drive_link)
  - [Altocumulus segmentation network](https://drive.google.com/file/d/15qBStqVLjUQDxQud4xSt4N8M4yhATtO5/view?usp=drive_link)
  - [Cirrocumulus segmentation network](https://drive.google.com/file/d/1cizT5bi7hcsXkqCJrDrqY7AfABfw0oPI/view?usp=drive_link)
- With trained model, you can infer multi-level MBD by using `predict_mask` function in `utils/seg_evaluation.py`
- `utils/seg_backbone.py`, `utils/seg_image_module.py`, `utils/seg_losses.py`, `utils/seg_model_utils.py`, `utils/seg_evaluation.py` contains codes for training semantic segmentation network. 

## 3) Module 2 : FS estimation from DL 
- Using preprocessed image and breast cancer status information, we trained [DenseNet121](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)
- Model was implemented with PyTorch
- Requirements are listed in `requirements/module2.txt`
  - Installation : `pip install -r module2.txt`
- Codes required for training model are in `utils/dl_model_utils.py`
  - You can train model with `main()` function.
- You can download trained networks in below link
  - [Model trained with Korean data](https://drive.google.com/file/d/12wBwl1tQ1R5bwoH1lIhrBLMtqzavSLJG/view?usp=drive_link)
  - [Model fine-tuned with EMBED white data](https://drive.google.com/file/d/1vzMrulTfQ1BsTX4vH-UFuv6ImiDKtzAz/view?usp=drive_link) 
 
## 4) Statistical Analysis
- Codes for statistical analysis (including plotting) are in `utils/stat_analysis.py`

## 5) XAI visualization 
- We applied Guided Grad-CAM to visualize trained model in module 2.
- Required codes are described in `utils/xai.py`
  - To use codes in this file, `gradcam.py`, `guided_backprop.py`, `guided_gradcam.py`, `misc_function.py`, `vanilla_backprop.py` files in `utils/cnn_vis` directory are required
- You can get Guided Grad-CAM visualization of last convolutional layer in trained model by using `gg_lastlayer` function in `utils/xai.py`
  

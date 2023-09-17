# The Brightest Area in Mammography coupled with Deep Learning Feature Scores: Identifying a High-Risk Breast Cancer Subgroup through Mammography Alone 

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

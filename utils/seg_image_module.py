import cv2
import os
import numpy as np
from tensorflow.keras.utils import Sequence

seg_image_dir = '' # Directory of input images 
seg_mask_1_dir = '' # Directory of input ground truth masks (Cumulus)
seg_mask_2_dir = '' # Directory of input ground truth masks (Altocumulus)
seg_mask_3_dir = '' # Directory of input ground truth masks (Cirrocumulus)

class DataGenerator(Sequence):
    
    def __init__(self, df, density, npy_col, split, batch_size, image_size, transform_both, transform_image_only):

        # df : Pandas Dataframe which has column for numpy path, data split, and etc. 
        # density : 1 for Cumulus / 2 for Altocumulus / 3 for Cirrocumulus 
        # npy_col : Name of column in df which contains path for input numpy arrays 
        # split : train / valid / test
        # transform_both : Transformation to apply both to input image and ground truth mask
        # transform_image_only : Transformation to apply only to input image (ex. Brightness modification)
        
        self.density = density # 1, 2, 3 (int)
        self.batch_size = batch_size
        self.split = split
        if self.split == "train":
            self.shuffle = True
            self.augmentation = True
        else:
            self.shuffle = False 
            self.augmentation = False
        
        self.image_size = image_size 
        self.transform_both = transform_both
        self.transform_image_only = transform_image_only
        self.df = df[df.split == split]
        self.npy_name_list = self.df[npy_col].tolist() 
        self.indices = list(range(len(self.npy_name_list)))
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, index):
        
        index_list = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = []
        y = []
        
        for idx in index_list:
            npy_name = self.npy_name_list[idx]
            image = np.load(os.path.join(seg_image_dir, npy_name))
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            if self.density == 1:
                mask = np.load(os.path.join(seg_mask_1_dir, npy_name))
            elif self.density == 2:
                mask = np.load(os.path.join(seg_mask_2_dir, npy_name))
            else: # 3
                mask = np.load(os.path.join(seg_mask_3_dir, npy_name))
            
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            
            if self.augmentation:
                transformed_both = self.transform_both(image=image, mask=mask)
                image = transformed_both["image"]
                mask = transformed_both["mask"]
                
                transformed_image = self.transform_image_only(image=image)
                image = transformed_image["image"]
            
            X.append(image)
            y.append(mask)
        
        X = np.array(X)
        X = X.reshape(-1, self.image_size, self.image_size, 1) 
        X = X.astype(np.float32)

        y = np.array(y)
        y = y.reshape(-1, self.image_size, self.image_size, 1) 
        y = y.astype(np.float32)

        gt_1 = y[:,::8,::8,:]
        gt_2 = y[:,::4,::4,:]
        gt_3 = y[:,::2,::2,:]
        gt_4 = y
        
        gt = [gt_1, gt_2, gt_3, gt_4]

        return X, gt
        
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
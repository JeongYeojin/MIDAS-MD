import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import sys
sys.path.append("..")
sys.path.append("../utils/cnn_vis")
from utils.cnn_vis.misc_functions import convert_to_grayscale
from utils.cnn_vis.vanilla_backprop import VanillaBackprop
from utils.cnn_vis.guided_backprop import GuidedBackprop
from utils.cnn_vis.gradcam import GradCam
from utils.cnn_vis.guided_gradcam import guided_grad_cam

import utils.dl_model_utils as dmu
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import cm
from skimage import transform
from scipy import ndimage
from skimage import io

def gg_lastlayer(model, image_path, device=torch.device("cuda")):

    '''
    Returns guided Grad-CAM of last convolutional layer of trained model 

    model : Trained deep learning model 
    image_path : Path of input image 
    '''
    
    img = np.load(image_path)
    img = img.astype(np.float32)
    img /= 255.0
    img = cv2.resize(img, (224, 224))
    img = img[np.newaxis, :, :]
    img = np.repeat(img, 3, 0) 
    img = img[np.newaxis, :, :, :]
    img = torch.from_numpy(img).to(device)
    
    pred = torch.sigmoid(model(img))
    
    nlayers = len(model.features._modules.items()) - 1
    
    grad_cam = GradCam(model, device, target_layer=nlayers-1)
    cam = grad_cam.generate_cam(img, 0)

    #GuidedBackprop
    GBP = GuidedBackprop(model, device)
    guided_grads = GBP.generate_gradients(img, 0)

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    
    return convert_to_grayscale(np.moveaxis(cam_gb,0,-1))

def img_gg_seg(npy_path, gg_npy_path, mask_1_path, mask_2_path, mask_3_path, smooth=True, threshold=0.5, visualize=True, legend=True):

    '''
    Function to plot original input image / Guided Grad-CAM / Predicted multi-level MBD 

    npy_path : Path of input image (numpy array)
    gg_npy_path : Path of Guided Grad-CAM of input image (generated from gg_lastlayer function)
    mask_1_path : Path of predicted binary mask corresponding to Cumulus 
    mask_2_path : Path of predicted binary mask corresponding to Altoumulus 
    mask_3_path : Path of predicted binary mask corresponding to Cirrocumulus 
    threshold : Threshold to apply to Guided Grad-CAM result
    '''
    
    img = np.load(npy_path) 
    gg = np.load(gg_npy_path) 
    mask_1 = np.load(mask_1_path) 
    mask_2 = np.load(mask_2_path) 
    mask_3 = np.load(mask_3_path) 
    
    # Apply gaussian filter 
    smooth_gg_b = gaussian_filter(gg.astype(float), sigma=1)
    smooth_gg_b /= smooth_gg_b.max()
    smooth_gg_b = np.where(smooth_gg_b > threshold, 1, 0)
    
    if smooth:
        img_list = [img, smooth_gg_b, seg[0, :, :], seg[1, :, :], seg[2, :, :]]
    else:
        img_list = [img, gg, seg[0, :, :], seg[1, :, :], seg[2, :, :]]
    title_list = ["Original", "Guided Grad-CAM", "Cumulus", "Altocumulus", "Cirrocumulus"]
    
    if visualize:
    
        fig, axes = plt.subplots(1, 5, figsize=(20,12))
        axes = axes.flatten()

        for i in range(5):
            ax = axes[i]
            if i != 1:
                ax.imshow(img_list[i], cmap="gray")
            else:
                im = ax.imshow(img_list[i], cmap="inferno") 
                axins = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=axins)
            if legend:
                ax.set_title(title_list[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    
    return img_list
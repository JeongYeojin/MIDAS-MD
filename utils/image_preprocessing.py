'''
Main reference : https://github.com/CleonWong/Can-You-Find-The-Tumour
'''


import numpy as np
import os
import cv2
import pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

IMAGE_SIZE = 256

def minMaxNormalise(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img

def sortContoursByArea(contours, reverse=True):

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes

def extractBreast(mask, top_x=None, reverse=True):
    
    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )
    
    n_contours = len(contours)
    
    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if n_contours < top_x or top_x == None:
            top_x = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )
        
    return n_contours, X_largest_blobs

def applyMask(img, mask, verbose=False):
    # mask : Binary image 
    # Applies binary mask to image to retain only desired areas 

    if verbose:
        print("img shape : ", img.shape)
        print("mask shape : ", mask.shape)
    
    masked_img = img.copy()
    try:
        masked_img[mask == 0] = 0
    except IndexError:
        print("Error: Dimensions of the mask do not match the image.")
        return None

    return masked_img

def checkLRFlip(img):

    # In preprocessing, we make all the mammograms to be on left.
    # Returns boolean : If True, the mammogram needs to be flipped

    # Get number of rows and columns in the image.
    nrows, ncols = img.shape
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = img.sum(axis=0)
    # Sum across each row.
    # row_sum = img.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False

    return LR_flip

def makeLRFlip(img):
    # Flip the image

    flipped_img = np.fliplr(img)

    return flipped_img

def flip_if_needed(img): 
    # Flip the image if needed 
    LR_flip = checkLRFlip(img)
    if LR_flip:
        return makeLRFlip(img)
    else:
        return img
    
def remove_left_border(img):
    
    # While removing peripheral skin, the left border also get eroded. So we additionally remove the eroded left border. 
    # Not needed for images used for binary classification (Since they do not undergo skin removal)

    nrows, ncols = img.shape 
    colSums = np.sum(img, axis=0)
    left_idx = np.min(np.where(colSums != 0)[0])
    cropped_img = img[:, left_idx:] 
    return cropped_img

def pad(img): 
    # Pad the remaining region to reshape the image to square 

    nrows, ncols = img.shape

    # If padding is required...
    if nrows != ncols:

        # Take the longer side as the target shape.
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)

        # pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

    # If padding is not required...
    elif nrows == ncols:

        # Return original image.
        padded_img = img

    return padded_img

def crop_img(img):
    
    # Additional function : Only retain the region with breast (not used in this study)
    
    nrows, ncols = img.shape
    colSums = np.sum(img, axis=0)
    rowSums = np.sum(img, axis=1)
    upper_idx = np.min(np.where(rowSums != 0)[0])
    lower_idx = np.max(np.where(rowSums != 0)[0])
    right_idx = np.max(np.where(colSums != 0)[0])
    cropped_img = img[upper_idx:lower_idx+1, :right_idx+1]
    
    return cropped_img

def load_dicom(dcm_path, voi_lut='SIGMOID'):
    
    '''
    Read DICOM -> Return DCM object / array / RescaleSlope / RescaleIntercept 
    voi_lut : 'SIGMOID' / 'LINEAR' ('SIGMOID' was used in this study)
    '''
    dcm = pydicom.read_file(dcm_path, force=True)
    
    if (0x0028, 0x1056) in list(dcm.keys()): # If DICOM header contains information about voi_lut 
        if dcm[0x0028, 0x1056].value == '':
            dcm[0x0028, 0x1056].value = 'SIGMOID' # Impute the value if empty 
    else: # Creates the information if DICOM header does not contain information about voi_lut 
        dcm.add_new((0x0028, 0x1056), 'LO', voi_lut)
    
    # Get window center and width information
    try:
        window_center = int(dcm.WindowCenter)
    except:
        window_center = int(dcm.WindowCenter[0])
    try:
        window_width = int(dcm.WindowWidth)
    except:
        window_width = int(dcm.WindowWidth[0])
    
    s = int(dcm.RescaleSlope)
    b = int(dcm.RescaleIntercept)
    
    # Get pixel array 
    try:
        arr = dcm.pixel_array
    except AttributeError:
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        arr = dcm.pixel_array
    except:
        dcm.decompress()
        arr = dcm.pixel_array
        
    return dcm, arr, s, b 

def dicom_to_img(dcm_path, windowing=True, voi_lut='SIGMOID'):
    
    # Return Windowed NumPy array from DICOM 
    
    dcm, arr, s, b = load_dicom(dcm_path, voi_lut='SIGMOID')
    
    img = s * arr + b
    
    if windowing:
        # Windowing 
        img = apply_modality_lut(img, dcm)
        img = apply_voi_lut(img, dcm)
    
    img = img - img.min()
    
    return img
    

def dicom_get_total_area(dcm_path, windowing=True, mask_thres=10, top_x=1, reverse=True, img_size=IMAGE_SIZE):
    
    # Returns total area of breast 
    '''
    Read DICOM -> Numpy Array -> Windowing
    '''
    img = dicom_to_img(dcm_path, windowing)
    
    '''
    Get Total Area Mask 
    '''
    mask_original = np.where(img > mask_thres, 1, 0)
    mask_original = np.uint8(mask_original)    
    
    '''
    Choose breast region from mask 
    '''
    _, mask_original = extractBreast(mask_original, top_x, reverse)
    
    mask_original = pad(mask_original)
    mask_original = np.uint8(mask_original)    
    
    resized_mask_original = cv2.resize(mask_original, dsize=(img_size, img_size))
    total_area = np.sum(resized_mask_original)
    
    return total_area



def preprocess_dicom(dcm_path, windowing=True, voi_lut='SIGMOID', mask_thres=10, min_max=True, top_x=1, reverse=True, img_size=IMAGE_SIZE, crop=False, apply_pad=True):
    
    ## Preprocess DICOM file including histogram window adjustment, flip, padding and resizing 
    ## Output : Preprocessed numpy array / Total area of breast

    # dcm_path : Path of DICOM file 
    # windowing=True -> Applies windowing function from PyDicom library (used in this study to alleviate inter-vendor difference)
    # voi_lut : VOI_LUT type used for windowing ('SIGMOID' or 'LINEAR')
    # mask_thres : Threshold of pixel value to make binary mask retaining breast region. 
        # Value may be differ according to vendor and other circumstances, but in this study 10 was appropriate for images after windwoing. 
    # top_x : number of contours to sort (default 1 -> choose only lagest (breast) region)
    # reverse=True : Sort contour area by descending order (Set True to retain the largest area)
    # img_size : Target image size 
    # crop=False : Do not apply crop 
    # apply_pad=True : Apply padding to make the resultant image square 
    
    '''
    Read DICOM -> Numpy Array
    '''
    img = dicom_to_img(dcm_path, windowing, voi_lut)
    
    '''
    Get Total Area Mask 
    '''
    mask = np.where(img > mask_thres, 1, 0) 
    mask = np.uint8(mask)    

    '''
    Min-Max Normalize
    '''
    if min_max:
        img = minMaxNormalise(img)
    else:
        img = img
    
    '''
    Choose breast region from mask 
    '''
    _, mask = extractBreast(mask, top_x, reverse)
    
    '''
    Apply mask to normalized img 
    '''
    res_img = applyMask(img, mask, verbose=False)
    
    '''
    Flip if needed
    '''
    res_img = flip_if_needed(res_img)
    res_mask = flip_if_needed(mask)
    
    '''
    Crop only region with breast (if needed)
    '''
    if crop:
        res_img = crop_img(res_img)
        res_mask = crop_img(res_mask)

    '''
    Pad
    '''
    if apply_pad:
        res_img = pad(res_img)
        res_mask = pad(res_mask)
    
    scaled_img = np.uint8(res_img * 255.0)
    res_mask = np.uint8(res_mask)
    
    resized_img= cv2.resize(scaled_img, dsize=(img_size, img_size))
    resized_mask = cv2.resize(res_mask, dsize=(img_size, img_size))
    
    total_area = np.sum(resized_mask)
    
    return resized_img, total_area 


    
    
def preprocess_dicom_seg(dcm_path, thres_1, thres_2, thres_3, windowing=True, voi_lut='SIGMOID', mask_thres=10, min_max=True, top_x=1, reverse=True, img_size=IMAGE_SIZE, border_thickness=10, \
                    erode_iter_num=5, crop=False, apply_pad=True):

    ## Preprocess DICOM file including histogram window adjustment, flip, padding and resizing + Skin removal 
    ## Output : Preprocessed numpy array / Corresponding ground truth mask for Cumulus, Altocumulus, and Cirrocumulus 

    # dcm_path : Path of DICOM file 
    # thres_1, thres_2, thres_3 : Expert annotated threshold for Cumulus, Altocumulus, and Cirrocumulus 
    # windowing=True -> Applies windowing function from PyDicom library (used in this study to alleviate inter-vendor difference)
    # voi_lut : VOI_LUT type used for windowing ('SIGMOID' or 'LINEAR')
    # mask_thres : Threshold of pixel value to make binary mask retaining breast region. 
        # Value may be differ according to vendor and other circumstances, but in this study 10 was appropriate for images after windwoing. 
    # top_x : number of contours to sort (default 1 -> choose only lagest (breast) region)
    # reverse=True : Sort contour area by descending order (Set True to retain the largest area)
    # img_size : Target image size 
    # border_thickness : Width of margin to remove from periphery of breast 
    # erode_iter_num : Number of iteration to call erosion (In this study, 3 ~ 5 was appropriate)
    # crop=False : Do not apply crop 
    # apply_pad=True : Apply padding to make the resultant image square 

    '''
    Load DICOM 
    ''' 
    dcm, arr, s, b = load_dicom(dcm_path, voi_lut)
    
    '''
    Windowing
    ''' 
    # bw : before windowing / aw : after windowing 
    img_bw = s * arr + b
    
    if windowing:
        img_bw = apply_modality_lut(img_bw, dcm) 
        img_aw = apply_voi_lut(img_bw, dcm)
    else:
        img_aw = img_bw.copy()
    img_aw = img_aw - img_aw.min() 
    
    mask_original = np.where(img_aw > mask_thres, 1, 0) # Binary mask 
    mask_original = np.uint8(mask_original)

    '''
    Min-Max Normalize
    '''
    if min_max:
        img_aw = minMaxNormalise(img_aw)
    else:
        img_aw = img_aw
    
    '''
    Choose breast region from mask and get total area
    '''
    _, mask_original = extractBreast(mask_original, top_x, reverse)
    
    mask_original_for_ta = pad(mask_original)
    mask_original_for_ta = np.uint8(mask_original_for_ta) # Used to calculate total area before erosion (skin removal)  
    
    resized_mask_original_for_ta = cv2.resize(mask_original_for_ta, dsize=(img_size, img_size))
    resized_mask_original_for_ta = np.uint8(resized_mask_original_for_ta)
    total_area = np.sum(resized_mask_original_for_ta)
    
    '''
    Remove skin from breast mask 
    '''
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (border_thickness, border_thickness))
    mask_eroded = cv2.erode(mask_original, kernel, borderValue=0, iterations=erode_iter_num)
    
    mask_eroded = np.uint8(mask_eroded)
    
    '''
    Apply mask to normalized img, so the periphery of breast are removed 
    We keep images before windowing to use expert annotated threshold to make ground truth mask (Since these thresholds are made based on original DICOM image)
    '''
    bw_img_eroded = applyMask(img_bw, mask_eroded, verbose=False)
    aw_img_eroded = applyMask(img_aw, mask_eroded, verbose=False)
    
    '''
    Flip if needed
    '''
    bw_img_eroded = flip_if_needed(bw_img_eroded)
    aw_img_eroded = flip_if_needed(aw_img_eroded)
    
    '''
    Remove left border (only for eroded image & mask)
    '''
    bw_img_eroded = remove_left_border(bw_img_eroded)
    aw_img_eroded = remove_left_border(aw_img_eroded)
    
    '''
    Crop only region with breast (if needed)
    '''
    if crop:
        bw_img_eroded = crop_img(bw_img_eroded)
        aw_img_eroded = crop_img(aw_img_eroded)
    
    '''
    Pad
    '''
    if apply_pad:
        bw_img_eroded = pad(bw_img_eroded)
        aw_img_eroded = pad(aw_img_eroded)
    
    '''
    MD generation & Scaling Images 
    '''
    # Multi-levle MD ground truth mask generation & Resizing 
    resized_density_1_eroded = cv2.resize(np.uint8(np.where(bw_img_eroded >= thres_1, 1, 0)), dsize=(img_size, img_size))
    resized_density_2_eroded = cv2.resize(np.uint8(np.where(bw_img_eroded >= thres_2, 1, 0)), dsize=(img_size, img_size))
    resized_density_3_eroded = cv2.resize(np.uint8(np.where(bw_img_eroded >= thres_3, 1, 0)), dsize=(img_size, img_size))
    
    # Image Scaling & Resizing
    scaled_img_eroded = np.uint8(aw_img_eroded * 255.0)
    resized_img_eroded = cv2.resize(scaled_img_eroded, dsize=(img_size, img_size))
    
    return resized_img_eroded, resized_density_1_eroded, resized_density_2_eroded, resized_density_3_eroded
    

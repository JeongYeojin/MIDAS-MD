import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_recall_curve

def dice_score(y_pred, y_true):
    
    smooth = 1
    pred = np.ndarray.flatten(np.clip(y_pred,0,1))
    gt = np.ndarray.flatten(np.clip(y_true,0,1))
    intersection = np.sum(pred * gt) 
    union = np.sum(pred) + np.sum(gt)   
    
    return np.round((2 * intersection + smooth)/(union + smooth),decimals=5)

def performance_metrics(y_true, y_pred):
    
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)
    
    recall = (tp + smooth) / (tp + fn + smooth) # recall
    specificity = (tn + smooth) / (tn + fp + smooth) # Specificity 
    precision = (tp + smooth) / (tp + fp + smooth) # precision
    
    return [recall, specificity, precision]
    
def eval_model(model, X_test, Y_test, thresh=0.5, visualize=True):

    # Evaluate semantic segmentation model 
    # X_test : Test input images
    # Y_test : Test input ground truth masks 
    # thresh : Threshold to make predicted binary mask 
    
    _, _, _, preds = model.predict(X_test)
    print("ground truth shape : ", X_test.shape) 
    print("prediction shape : ", preds.shape) 
    
    dice_score_list = np.zeros((len(X_test), 1))
    recall_list = np.zeros_like(dice_score_list)
    specificity_list = np.zeros_like(dice_score_list)
    precision_list = np.zeros_like(dice_score_list)
    
    for i in range(len(X_test)):
        
        gt_mask = Y_test[i:i+1]
        pred_mask = preds[i:i+1] 
        dice_score_list[i] = dice_score(pred_mask >= thresh, gt_mask)
        
        rc, sp, prc = performance_metrics(gt_mask, pred_mask)
        recall_list[i] = rc
        specificity_list[i] = sp
        precision_list[i] = prc
        
    print('-'*30)
    print('USING HDF5 saved model at thresh=', str(thresh))
    
    print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
        np.sum(dice_score_list)/len(X_test),  
        np.sum(recall_list)/len(X_test),
        np.sum(precision_list)/len(X_test)))
    
    if visualize:
        # plot precision-recall 
        y_true = Y_test.ravel() 
        y_preds = preds.ravel() 
        precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
        plt.figure(20)
        plt.plot(recall,precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision") 
        plt.show() 
        
def predict_mask(npy_path, model_1, model_2, model_3, IMAGE_SIZE):

    # Make prediction of cumulus, altocumulus, cirrocumulus using input image
    # npy_path : Path of numpy file to predict 
    # model_1 : Trained semantic segmentation model using Cumulus as ground truth 
    # model_2 : Trained semantic segmentation model using Altocumulus as ground truth 
    # model_3 : Trained semantic segmentation model using Cirrocumulus as ground truth 
    
    image = np.load(npy_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    image = image.astype(np.float32)
    
    _, _, _, predicted_mask_1 = model_density_1.predict(image)
    _, _, _, predicted_mask_2 = model_density_2.predict(image)
    _, _, _, predicted_mask_3 = model_density_3.predict(image)
    
    predicted_mask_1 = (predicted_mask_1 > 0.5).astype(np.uint8)
    predicted_mask_2 = (predicted_mask_2 > 0.5).astype(np.uint8)
    predicted_mask_3 = (predicted_mask_3 > 0.5).astype(np.uint8)
    
    predicted_density_1 = predicted_mask_1[0][:, :, 0]
    predicted_density_2 = predicted_mask_2[0][:, :, 0]
    predicted_density_3 = predicted_mask_3[0][:, :, 0]

    return predicted_density_1, predicted_density_2, predicted_density_3

    

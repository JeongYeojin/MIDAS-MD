import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, recall_score, precision_score
from tqdm import tqdm
from pytorch_optimizer import Ranger
import torch
from torchvision.models import densenet121
import albumentations as A


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms, channel_num=1):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms
        self.channel_num = channel_num
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        
        data, label = self.image_files[index], self.labels[index]
        img = np.load(data)
        img = img.astype(np.float32)
        img /= 255.0
        
        if self.transforms:
            augmentations = self.transforms(image=img)
            img = augmentations["image"]
            img = img[np.newaxis, :, :]
            
        label = torch.from_numpy(np.array(label))
        img = torch.from_numpy(img)
            
        return img, label

def get_model():

    model = densenet121(pretrained=False, num_classes=1)
    return model 

def get_optimizer(model, learning_rate):

    optimizer = Ranger(model.parameters(lr=learning_rate))
    return optimizer 

def get_loss_function(pos_weight):

    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_function 
        

def get_dataloader(df, channel_num, target_var, image_col, image_size, batch_size):

    # In pandas DataFrame df, it has column named "split" - "train", "valid", "test"
    # df : Pandas dataframe containing information
    # target_var : Name of column in df correspondin to breast cancer status
    # image_col : Name of column in df corresponding to input image (numpy array) path
    
    train_transforms = A.Compose([
            A.Resize(width=image_size, height=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])

    val_transforms = A.Compose([ 
                A.Resize(width=image_size, height=image_size, p=1.0),
            ])
    
    X_train = df[df["split"] == "train"][image_col].tolist()
    X_valid = df[df["split"] == "valid"][image_col].tolist()
    X_test = df[df["split"] == "test"][image_col].tolist()

    Y_train = df[df["split"] == "train"][target_var].tolist()
    Y_valid = df[df["split"] == "valid"][target_var].tolist()
    Y_test = df[df["split"] == "test"][target_var].tolist()

    train_ds = CustomDataset(X_train, Y_train, train_transforms, channel_num=channel_num)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    valid_ds = CustomDataset(X_valid, Y_valid, val_transforms, channel_num=channel_num)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    test_ds = CustomDataset(X_test, Y_test, val_transforms, channel_num=channel_num)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    
    return train_ds, train_loader, valid_ds, valid_loader, test_ds, test_loader
    
def train(model, hist_name, model_dir, train_loader, valid_loader, metric_list, device, optimizer, loss_function, binary_threshold, max_epochs=100):

    # model
    # hist_name : Name of csv file to store performance of model in each epoch
    # model_dir : Directory to save the model 
    # metric_list : List of validation metric to monitor 

    hist_path = os.path.join(model_dir, f"{hist_name}.csv") # Record model performance of each epoch  
    
    best_loss = 10000
    best_acc = -1
    best_auprc = -1
    best_auroc = -1
    best_recall = -1
    best_precision = -1
    epoch_train_loss_values = []
    epoch_valid_loss_values = []
    epoch_valid_acc_values = []
    epoch_valid_auprc_values = []
    epoch_valid_auroc_values = []
    epoch_valid_recall_values = []
    epoch_valid_precision_values = []
    
    hist_df = pd.DataFrame({"train_loss" : [], "valid_loss" : [], "valid_acc" : [], "valid_auprc" : [], "valid_auroc" : [], "valid_recall" : [], "valid_precision" : []})

    for epoch in range(0, max_epochs):
        
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_train_loss = 0
        epoch_valid_loss = 0
        train_step = 0
        valid_step = 0

        for train_data in tqdm(train_loader):
            train_step += 1
            train_X, train_y = train_data[0].to(device),train_data[1].to(device)
            train_y = train_y.float()
            optimizer.zero_grad()
            train_y_pred = model(train_X)
            train_loss = loss_function(train_y_pred, train_y.unsqueeze(1))
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()

        epoch_train_loss /= train_step
        epoch_train_loss_values.append(epoch_train_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_train_loss:.4f}")

        ## Validation
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.float32, device=device)
            for valid_data in valid_loader:
                valid_step += 1
                valid_X, valid_y = valid_data[0].to(device), valid_data[1].to(device)
                valid_y_pred = model(valid_X)
                valid_y_pred = torch.squeeze(valid_y_pred, 1)
                y_pred = torch.cat([y_pred, valid_y_pred.float()], dim=0)
                y = torch.cat([y, valid_y], dim=0)

                valid_loss = loss_function(valid_y_pred, valid_y.float())
                epoch_valid_loss += valid_loss.item() 

            epoch_valid_loss /= valid_step
            epoch_valid_loss_values.append(epoch_valid_loss)

            y_pred_sigmoid = torch.sigmoid(y_pred) # sigmoid prediction 
            y_pred_binary = torch.where(y_pred_sigmoid >= binary_threshold, 1, 0).to(device) # Sigmoid prediction to Binary prediction

            y_cpu = y.cpu()
            y_pred_sigmoid_cpu = y_pred_sigmoid.cpu()
            y_pred_binary_cpu = y_pred_binary.cpu()

            precision, recall, _ = precision_recall_curve(y_cpu, y_pred_sigmoid_cpu)
            epoch_valid_auprc = auc(recall, precision)
            epoch_valid_recall = recall_score(y_cpu, y_pred_binary_cpu)
            epoch_valid_precision = precision_score(y_cpu, y_pred_binary_cpu)
            epoch_valid_auroc = roc_auc_score(y_cpu, y_pred_sigmoid_cpu)

            epoch_valid_auprc_values.append(epoch_valid_auprc)
            epoch_valid_auroc_values.append(epoch_valid_auroc)
            epoch_valid_recall_values.append(epoch_valid_recall)
            epoch_valid_precision_values.append(epoch_valid_precision)

            acc_value = torch.eq(y_pred_binary, y)
            epoch_valid_acc = acc_value.sum().item() / len(acc_value)
            epoch_valid_acc_values.append(epoch_valid_acc)

            print(f"epoch {epoch + 1} average valid loss: {epoch_valid_loss:.4f}, average valid acc : {epoch_valid_acc:.4f}, valid AUROC : {epoch_valid_auroc:.4f}, valid AUPRC : {epoch_valid_auprc:.4f}, valid recall : {epoch_valid_recall:.4f}, valid precision : {epoch_valid_precision:.4f}")

            if "loss" in metric_list:
                if epoch_valid_loss < best_loss:
                    best_loss = epoch_valid_loss
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_loss.pth"))
            if "accuracy" in metric_list:
                if epoch_valid_acc > best_acc:
                    best_acc = epoch_valid_acc
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_acc.pth"))
            if "auroc" in metric_list:
                if epoch_valid_auroc > best_auroc:
                    best_auroc = epoch_valid_auroc
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_auroc.pth"))        
            if "auprc" in metric_list:
                if epoch_valid_auprc > best_auprc:
                    best_auprc = epoch_valid_auprc
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_auprc.pth"))        
            if "recall" in metric_list:
                if epoch_valid_recall > best_recall:
                    best_recall = epoch_valid_recall
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_recall.pth"))
            if "precision" in metric_list:
                if epoch_valid_precision > best_precision:
                    best_precision = epoch_valid_precision
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{hist_name}_precision.pth"))          

        hist_df.loc[epoch] = [epoch_train_loss, epoch_valid_loss, epoch_valid_acc, epoch_valid_auprc, epoch_valid_auroc, epoch_valid_recall, epoch_valid_precision]
        hist_df.to_csv(hist_path, header=True, index=False)


def main(df, target_var, channel_num, learning_rate, metric_list, binary_threshold, image_size, batch_size, model_dir, hist_name):
    
    # df : Pandas DataFrame which has column for image path, breast cancer phenotype, and etc. 
    # target_var : Name of column in dataframe df which has breast cancer phenotype
    # hist_name : Name of csv file to store performance of model in each epoch
    # model_dir : Directory to save the model 

    train_ds, train_loader, valid_ds, valid_loader, test_ds, test_loader = get_dataloader(df, channel_num=channel_num, image_size=image_size, batch_size=batch_size)
    model = get_model()
    
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, device_ids = [0, 1, 2]) # Change according to your GPU setting 

    model = model.to(device)
    
    case_num = df[target_var].value_counts().get(1, 0) # Total number of breast cancer cases 
    control_num = df[target_var].value_counts().get(0, 0) # Total number of controls 
    weight_positive = control_num / case_num # Weight for binary cross entropy 
    pos_weight = torch.tensor([weight_positive]).to(device)

    loss_function = get_loss_function(pos_weight)
    loss_function.cuda()
    
    optimizer = get_optimizer(model, learning_rate=learning_rate)
    
    train(model, hist_name, model_dir, train_loader, valid_loader, metric_list, device, optimizer, loss_function, binary_threshold, max_epochs=100)

def load_model(df, pth_path, target_var, channel_num, image_size, batch_size, model_dir, to_device=True):

    # Load trained model
    # pth_path : path of pth file (weights) of trained model 
    # target_var : Name of column in df correspondin to breast cancer status

    model = get_model()
    
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, device_ids = [0, 1, 2]) # Change according to your GPU setting 
    model = model.to(device)
    
    case_num = df[target_var].value_counts().get(1, 0)
    control_num = df[target_var].value_counts().get(0, 0)
    weight_positive = control_num / case_num
    pos_weight = torch.tensor([weight_positive]).to(device)
    
    loss_function = get_loss_function(pos_weight)
    loss_function.cuda()
    
    # Load trained model 
    model.load_state_dict(torch.load(pth_path))

    return model 
    
def test(df, pth_path, channel_num, image_size, batch_size, binary_threshold, model_dir, visualize, figsize=(8,6), fontsize=12, label_fontsize=12, legend=True):

    # pth_path : Path of pth file (weights) of trained model
    _, _, _, _, _, test_loader = get_dataloader(df, channel_num=channel_num, image_size=image_size, batch_size=batch_size)

    model = get_model()
    
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, device_ids = [0, 1, 2])
    model = model.to(device)
    
    case_num = df[target_var]..value_counts().get(1, 0)
    control_num = df[target_var]..value_counts().get(0, 0)
    weight_positive = control_num / case_num
    pos_weight = torch.tensor([weight_positive]).to(device)
    
    loss_function = get_loss_function(pos_weight)
    loss_function.cuda()
    
    # Load trained model 
    model.load_state_dict(torch.load(pth_path))

    test_step = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.float32, device=device)
        for test_data in test_loader:
            test_step += 1
            test_X, test_y = test_data[0].to(device), test_data[1].to(device)
            test_y_pred = model(test_X)
            test_y_pred = torch.squeeze(test_y_pred, 1)
            y_pred = torch.cat([y_pred, test_y_pred.float()], dim=0)
            y = torch.cat([y, test_y], dim=0)
            test_loss_item = loss_function(test_y_pred, test_y.float())
            test_loss += test_loss_item.item()
        
        test_loss /= test_step
        
        y_pred_sigmoid = torch.sigmoid(y_pred) # sigmoid prediction 
        y_pred_binary = torch.where(y_pred_sigmoid >= binary_threshold, 1, 0).to(device) # Sigmoid prediction to Binary prediction

        y_cpu = y.cpu()
        y_pred_sigmoid_cpu = y_pred_sigmoid.cpu()
        y_pred_binary_cpu = y_pred_binary.cpu()

        precision, recall, _ = precision_recall_curve(y_cpu, y_pred_sigmoid_cpu)
        test_auprc = auc(recall, precision)
        test_recall = recall_score(y_cpu, y_pred_binary_cpu)
        test_precision = precision_score(y_cpu, y_pred_binary_cpu)
        test_auroc = roc_auc_score(y_cpu, y_pred_sigmoid_cpu)

        acc_value = torch.eq(y_pred_binary, y)
        test_acc = acc_value.sum().item() / len(acc_value)
        
        print(f"Model name : {pth_name} \n Loss {test_loss} \n Accuracy {test_acc} \n AUROC {test_auroc} \n AUPRC {test_auprc} \n Recall {test_recall} \n Precision {test_precision}")
        print(f"Confusion matrix : \n {confusion_matrix(y_cpu, y_pred_binary_cpu)}")
        
    if visualize:
        
        plt.figure(figsize=figsize)
        fpr, tpr, _ = roc_curve(y_cpu, y_pred_sigmoid_cpu)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f"Feature Score (AUROC = {roc_auc:.3f})")

        # Plot the diagonal line representing random guessing
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

        # Set labels and title
        if legend:
            plt.xlabel("False Positive Rate", fontsize=fontsize)
            plt.ylabel("True Positive Rate", fontsize=fontsize)
        plt.xticks([])  
        plt.yticks([])  

        # Add legend
        plt.legend(loc="lower right", prop={'size': label_fontsize})

        # Show the plot
        plt.show()


    

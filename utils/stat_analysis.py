import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import yeojohnson
from scipy import stats
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LinearRegression, LogisticRegression


def density_adjustment(df, density_var_1, density_var_2, density_var_3, x_var_list, adjusted_density_var_name_1, adjusted_density_var_name_2, adjusted_density_var_name_3):
    
    '''
    Function for density adjustment (Described in paper)
    Adjust pMBD (percent of MBD to total area) using age and total area 

    Arguments 
    df : Pandas dataframe containing information
    density_var_1 : Name of column corresponding to Cumulus MBD 
    density_var_2 : Name of column corresponding to Altocumulus MBD 
    density_var_3 : Name of column corresponding to Cirrocumulus MBD 
    x_var_list : List of variables to adjust (In this paper, age and total area)
    adjusted_density_var_name_1 : Name of column corresponding to adjusted Cumulus MBD 
    adjusted_density_var_name_2 : Name of column corresponding to adjusted Altocumulus MBD 
    adjusted_density_var_name_3 : Name of column corresponding to adjusted Cirrocumulus MBD 
    '''

    # Step 1 : Log Transformation
    
    df[f"{density_var_1}_tr"] = np.log(df[density_var_1] + 1)
    df[f"{density_var_2}_tr"] = np.log(df[density_var_2] + 1)
    df[f"{density_var_3}_tr"] = np.log(df[density_var_3] + 1)
    
    # Step 2 : Calculate the residuals by performing linear regression
    
    model_1 = LinearRegression()
    model_1.fit(df[x_var_list], df[f"{density_var_1}_tr"])
    model_2 = LinearRegression()
    model_2.fit(df[x_var_list], df[f"{density_var_2}_tr"])
    model_3 = LinearRegression()
    model_3.fit(df[x_var_list], df[f"{density_var_3}_tr"])
    
    density_1_predicted = model_1.predict(df[x_var_list])
    density_2_predicted = model_2.predict(df[x_var_list])
    density_3_predicted = model_3.predict(df[x_var_list])
    
    density_1_residuals = df[f"{density_var_1}_tr"] - density_1_predicted
    density_2_residuals = df[f"{density_var_2}_tr"] - density_2_predicted
    density_3_residuals = df[f"{density_var_3}_tr"] - density_3_predicted
    
    # Step 3 : Apply Yeo-Johnson transformation to make the residuals normally distributed
    
    density_1_transformed_residuals, density_1_optimal_lambda = yeojohnson(density_1_residuals)
    density_2_transformed_residuals, density_2_optimal_lambda = yeojohnson(density_2_residuals)
    density_3_transformed_residuals, density_3_optimal_lambda = yeojohnson(density_3_residuals)
    
    df[adjusted_density_var_name_1] = density_1_transformed_residuals
    df[adjusted_density_var_name_2] = density_2_transformed_residuals
    df[adjusted_density_var_name_3] = density_3_transformed_residuals
    
    df.drop(columns=[f"{density_var_1}_tr", f"{density_var_2}_tr", f"{density_var_3}_tr"], inplace=True) # Remove unnecessary columns 

    return df

def compare_distribution(df, x, y, group_order, figsize, x_name, y_name, left_shape, right_shape, fontsize, label=True):
    
    '''
    Function to plot histogram and violin (or box) plot to compare distribution of variables according to breast cancer status 

    df : Pandas variabe
    x : Name of column in df corresponding to variable to compare
    y : Name of column in df corresponding to breast cancer status 
    group_order : List of label (used for plot)
    x_name : Name for x-axis in left plot 
    left_shape : Type of plot in left. "bar" or "kde" 
    right_shape : Type of plot in right. "Violin" or "Box" 
    '''
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1) Histogram 
    
    if left_shape == "bar":
        hist_ax = sns.histplot(data=df, x=x, hue=y, hue_order=group_order, stat="density", common_norm=False, bins=30, kde=True, ax=axes[0], element="step")
    else:
        hist_ax = sns.kdeplot(data=df, x=x, hue=y, hue_order=group_order, common_norm=False, ax=axes[0])
    
    # Remove the legend
    hist_ax.get_legend().remove()
    
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    
    if not label:
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
    else:
        axes[0].set_xlabel(x_name)
        axes[0].set_ylabel("Frequency")
    
    # 2) Violin Plot 
    
    if right_shape == "violin":
        sns.violinplot(x=y, y=x, data=df, ax=axes[1])
    else:
        sns.boxplot(x=y, y=x, data=df, ax=axes[1], showfliers=False, order=group_order)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    if not label:
        axes[1].set_xticklabels('')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    

def MD_fig_2(df, MD_var_1, MD_var_2, MD_var_3, DL_var, target_var, figsize, fontsize):

    '''
    Function to show 3 x 3 OR tables according to tertiles of pMBD and FS 

    df : Pandas dataframe containing informations
    MD_var_1 : Name of column in df corresponding to adjusted Cumulus pMBD 
    MD_var_2 : Name of column in df corresponding to adjusted Altoumulus pMBD 
    MD_var_3 : Name of column in df corresponding to adjusted Cirrocumulus pMBD 
    DL_var : Name of column in df corresponding to FS 
    target_var : Name of column in df corresponding to breast cancer status
    '''
    
    # Matrix for Cumulus
    case_matrix_1 = np.zeros(shape=(3, 3))
    control_matrix_1 = np.zeros(shape=(3, 3))
    or_matrix_1 = np.zeros(shape=(3, 3))
    
    # Matrix for Altocumulus
    case_matrix_2 = np.zeros(shape=(3, 3))
    control_matrix_2 = np.zeros(shape=(3, 3))
    or_matrix_2 = np.zeros(shape=(3, 3))
    
    # Matrix for Cirrocumulus
    case_matrix_3 = np.zeros(shape=(3, 3))
    control_matrix_3 = np.zeros(shape=(3, 3))
    or_matrix_3 = np.zeros(shape=(3, 3))
    
    df["Y_q"]= pd.qcut(df[DL_var], 3, labels=False)
    df["X_1_q"]= pd.qcut(df[MD_var_1], 3, labels=False)
    df["X_2_q"]= pd.qcut(df[MD_var_2], 3, labels=False)
    df["X_3_q"]= pd.qcut(df[MD_var_3], 3, labels=False)

    for i in range(3):
        for j in range(3):
            df_cell_1 = df[(df["X_1_q"] == j) & (df["Y_q"] == 2-i)]
            df_cell_2 = df[(df["X_2_q"] == j) & (df["Y_q"] == 2-i)]
            df_cell_3 = df[(df["X_3_q"] == j) & (df["Y_q"] == 2-i)]
            cell_1_case_num = df_cell_1[target_var].value_counts().get(1,0)
            cell_2_case_num = df_cell_2[target_var].value_counts().get(1,0)
            cell_3_case_num = df_cell_3[target_var].value_counts().get(1,0)
            cell_1_control_num = df_cell_1[target_var].value_counts().get(0,0)
            cell_2_control_num = df_cell_2[target_var].value_counts().get(0,0)
            cell_3_control_num = df_cell_3[target_var].value_counts().get(0,0)

            case_matrix_1[i][j] = cell_1_case_num
            case_matrix_2[i][j] = cell_2_case_num
            case_matrix_3[i][j] = cell_3_case_num
            control_matrix_1[i][j] = cell_1_control_num
            control_matrix_2[i][j] = cell_2_control_num
            control_matrix_3[i][j] = cell_3_control_num
                
    df_mid_row = df[df["Y_q"] == 1]
    base_case_num = df_mid_row[target_var].value_counts().get(1,0)
    base_control_num = df_mid_row[target_var].value_counts().get(0,0)
    for i in range(3):
        for j in range(3):
            or_matrix_1[i][j] = (case_matrix_1[i][j] * base_control_num) / (base_case_num * control_matrix_1[i][j])
            or_matrix_2[i][j] = (case_matrix_2[i][j] * base_control_num) / (base_case_num * control_matrix_2[i][j])
            or_matrix_3[i][j] = (case_matrix_3[i][j] * base_control_num) / (base_case_num * control_matrix_3[i][j])
            
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    heatmap_1 = axes[0].imshow(or_matrix_1, cmap='coolwarm', interpolation='nearest')
        
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])

    for i in range(or_matrix_1.shape[0]):
        for j in range(or_matrix_1.shape[1]):
            axes[0].text(j, i, f'{or_matrix_1[i, j]:.2f} \n Case {int(case_matrix_1[i, j])} \n Control {int(control_matrix_1[i, j])}', ha='center', va='center', color='white', fontsize=fontsize)

    axes[0].set_xlabel(MD_var_1, fontsize=fontsize)
    axes[0].set_ylabel(DL_var, fontsize=fontsize)
    
    heatmap_2 = axes[1].imshow(or_matrix_2, cmap='coolwarm', interpolation='nearest')
        
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])

    for i in range(or_matrix_2.shape[0]):
        for j in range(or_matrix_2.shape[1]):
            axes[1].text(j, i, f'{or_matrix_2[i, j]:.2f} \n Case {int(case_matrix_2[i, j])} \n Control {int(control_matrix_2[i, j])}', ha='center', va='center', color='white', fontsize=fontsize)

    axes[1].set_xlabel(MD_var_2, fontsize=fontsize)
    axes[1].set_ylabel(DL_var, fontsize=fontsize)
    
    heatmap_3 = axes[2].imshow(or_matrix_3, cmap='coolwarm', interpolation='nearest')
        
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])

    for i in range(or_matrix_3.shape[0]):
        for j in range(or_matrix_3.shape[1]):
            axes[2].text(j, i, f'{or_matrix_3[i, j]:.2f} \n Case {int(case_matrix_3[i, j])} \n Control {int(control_matrix_3[i, j])}', ha='center', va='center', color='white', fontsize=fontsize)

    axes[2].set_xlabel(MD_var_3, fontsize=fontsize)
    axes[2].set_ylabel(DL_var, fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()
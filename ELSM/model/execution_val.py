# coding=gb2312
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier   
import numpy as np   
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader  
import sys
import torch 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import random
import itertools
import math
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.preprocessing import StandardScaler  
from torch.utils.data import random_split
from sklearn.model_selection import GridSearchCV
from torch.utils.data import  TensorDataset
from itertools import product
import torch.nn as nn
from torch.utils.data import  Subset
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse


from validate import validate_val
from train import train_model_val
from ELSM import ELSM
from data_processing import get_val_datasetlist_execution,CustomDataset,creat_dataloader_exe 
 


def fix_all_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except NameError:
        pass


def calculate_accuracy_and_confusion(result_df_1):
    def calculate_auc(result_df):  
        labels = result_df['original_labels']
        probs = result_df['probs']
        
        auc = roc_auc_score(labels, probs)
        return auc
    auc_1 = calculate_auc(result_df_1)
    return auc_1
    

# Hyperparameter search space
param_grid = {
    'learning_rate': [0.001, 0.01,0.1],
    'hidden_layer_size': [50, 100,110,120,130,140, 150, 200],  
    'num_epochs': [10, 20, 30,50,100]  
}

'''
This function performs 10-fold cross-validation on the training data for hyperparameter optimization, 
trains and evaluates the model, and returns the average AUC across all folds.
'''

def evaluate_model(params,TRAIN_data,modal_input_sizes,modal_input_sizes2,samples_sizes): 
    TRAIN_data1 = [np.unique(matrix, axis=0) for matrix in TRAIN_data]
    dataset = CustomDataset(TRAIN_data1)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    auc_list = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"Fold {fold+1}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        
        model = ELSM(modal_input_sizes, modal_input_sizes2, 
                                params['hidden_layer_size'], 2, samples_sizes)
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=params['learning_rate'], weight_decay=1e-5)
        
        train_model_val(model, nn.CrossEntropyLoss(), optimizer, train_loader, 
                    num_epochs=params['num_epochs'])
        
        final_df = validate_val(val_loader, model)
        auc = calculate_accuracy_and_confusion(final_df) 
        auc_list.append(auc)
    
    avg_auc = sum(auc_list) / len(auc_list)
    return avg_auc


# python3 execution_val.py "../sample_level_evaluation_strategy_result/Independent_Validation/"
def main():
    parser = argparse.ArgumentParser(description='Sample-level Evaluation Strategy')
    parser.add_argument('string1', type=str)
    args = parser.parse_args()
    print(f"input path : {args.string1}")
    input_path = args.string1


    TRAIN_data = [None for _ in range(13)]  
    TEST_data = [None for _ in range(13)]
    ALL_data = [None for _ in range(13)]
    
    train_df_list,test_df_list = get_val_datasetlist_execution(input_path)
    fix_all_random_seeds(44)
    pp = 0
    for filedata1,filedata2 in zip(train_df_list,test_df_list):
        train_loader,test_loader,all_loader= creat_dataloader_exe(filedata1,filedata2,44)
        TRAIN_data[pp] = train_loader
        TEST_data[pp] = test_loader
        ALL_data[pp] = all_loader
        pp = pp+1
    
    
    
    modal_input_sizes = [df.shape[1] - 3 for df in train_df_list]
    modal_input_sizes2 = [size // 2 for size in modal_input_sizes]
    samples_sizes = [df.shape[0] for df in train_df_list]
    TRAIN_data = [df.to_numpy() for df in TRAIN_data]
    TEST_data = [df.to_numpy() for df in TEST_data]
    
    
    best_params = {
        'learning_rate': 0.01, 
        'hidden_layer_size': 100,  
        'num_epochs': 30 
    }
    best_auc = 0.0
    
    
    model = ELSM(modal_input_sizes, modal_input_sizes2 ,best_params['hidden_layer_size'] , 2,samples_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
    
    dataset = CustomDataset(TRAIN_data)  
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    train_model_val(model, criterion, optimizer, train_loader, num_epochs=best_params['num_epochs'])   
    
    dataset_test = CustomDataset(TEST_data)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True)    
    
    final_df_1 = validate_val(test_loader , model)        
    auc_1= calculate_accuracy_and_confusion(final_df_1)
    print(auc_1)
    print(final_df_1)

    final_df_1.to_csv("../result/val_df.csv", index=False)
  


 
if __name__ == "__main__":
    main()
        
        

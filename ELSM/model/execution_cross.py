# coding=gb2312 
import os
import sys
import math
import glob
import random
import itertools
from collections import Counter
import argparse

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)
import torch.optim as optim 
import torch    
from torch.utils.data import Dataset, DataLoader ,Subset 
from itertools import product
import torch.nn as nn  
import torch.nn.functional as F  


from ELSM import ELSM
from data_processing import read_test_csv_execross,read_train_csv_execross,creat_dataloader_execross,CustomDataset_execross
from train import train_model_cross
from validate import validate_cross




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
        labels = result_df['labels']
        labels = result_df['labels'].replace({2: 0, 3: 0})
        
        probs = result_df['score']
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

def evaluate_model(params):
    TRAIN_data1 = [np.unique(matrix, axis=0) for matrix in TRAIN_data]
    dataset = CustomDataset_execross(TRAIN_data1)
    kfold = KFold(n_splits=10, shuffle=True, random_state=44)
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
        
        train_model_cross(model, nn.CrossEntropyLoss(), optimizer, train_loader, 
                    num_epochs=params['num_epochs'])
        
        final_df,_ = validate_cross(val_loader, model)
        
        
        auc = calculate_accuracy_and_confusion(final_df) 
        auc_list.append(auc)
    avg_auc = sum(auc_list) / len(auc_list)
    return avg_auc




# python3 execution_cross.py "../sample_level_evaluation_strategy_result/"
def main():
    parser = argparse.ArgumentParser(description='Sample-level Evaluation Strategy')
    parser.add_argument('string1', type=str)
    args = parser.parse_args()
    print(f"input path : {args.string1}")
    path = args.string1
    fix_all_random_seeds(44)
    
    PPPP=0
    all_big_dfs = []
    while(PPPP<=4):
        PPPPP=0
        while(PPPPP<10):  
            dirname=f"10-fold-cross-validation_{PPPP+1}_{PPPPP}"    
            print(dirname) 
            
            #path = "../sample_level_evaluation_strategy_result/"
            
            train_df_list = read_train_csv_execross(dirname,path)
            test_df_list = read_test_csv_execross(dirname,path)
        
            TRAIN_data = [None for _ in range(13)]  
            TEST_data = [None for _ in range(13)]
            ALL_data = [None for _ in range(13)]  
            
            pp =0 
            for filedata1,filedata2 in zip(train_df_list,test_df_list):
                train_loader,test_loader,all_loader = creat_dataloader_execross(filedata1,filedata2)
                TRAIN_data[pp] = train_loader
                TEST_data[pp] = test_loader
                ALL_data[pp] = all_loader
                pp = pp+1
        
            modal_input_sizes = [df.shape[1] - 3 for df in train_df_list]    
            modal_input_sizes2 = [max(size // 2, 10) for size in modal_input_sizes]
            
            samples_sizes = [df.shape[0] for df in train_df_list]
    
    
    
            best_params = {
                'learning_rate': 0.001, 
                'hidden_layer_size': 130,  
                'num_epochs': 10  
            }
            best_auc = 0.0
            

  
            
            model = ELSM(modal_input_sizes, modal_input_sizes2 ,best_params['hidden_layer_size']  , 2,samples_sizes)
            
            TRAIN_data1 = [df.to_numpy() for df in TRAIN_data]
            TEST_data1 = [df.to_numpy() for df in TEST_data]
        
            #TRAIN
            criterion = nn.CrossEntropyLoss()  
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
            
            dataset = CustomDataset_execross(TRAIN_data1)  
            train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
            train_model_cross(model, criterion, optimizer, train_loader, num_epochs=best_params['num_epochs'])  
            
            #TEST
            dataset_test = CustomDataset_execross(TEST_data1)
            test_loader = DataLoader(dataset_test, batch_size=64, shuffle=True)  
            big_df,auc = validate_cross(test_loader , model)    
            print(auc)
            all_big_dfs.append(big_df) 
    
            PPPPP+=1
        PPPP+=1
    final_df = pd.concat(all_big_dfs, ignore_index=True)
    
    
    final_df.to_csv("../result/cross_df.csv", index=False)

 
if __name__ == "__main__":
    main() 


# coding=gb2312 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
import random
import itertools
import math
from collections import Counter
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score 
import torch.optim as optim  
import argparse

from data_processing import creat_dataloader,CustomDataset
from ELSM import ELSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = [  
    'coverage',
    'delfi' , 
    'end',  
    'end_motifs',  
    'fragment',  
    'fragmenter',  
    'FSD',  
    'FSR', 
    'IFS', 
    'length',  
    'WPS',  
    'OCF',  
    'PFE'  
]

def fix_all_random_seeds(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except NameError:
        pass
      
def train_model(model, criterion, optimizer,train_loader, num_epochs): 
    model = model.to(device)
    for epoch in range(num_epochs): 
        model.train()
        total_loss = 0   
        for inputs, labels,period in train_loader:    
            inputs = [input_.float().to(device) for input_ in inputs] 
            labels = labels.long()
            labels = labels.to(device)

            optimizer.zero_grad()   
            outputs = model(inputs)  
            outputs = outputs.to(device)
            
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()
            total_loss += loss.item() 
        avg_loss = total_loss / len(train_loader)  
        torch.cuda.empty_cache()
 
def validate(val_loader, model):    
    model = model.to(device)
    model.eval()   
    all_preds = []  
    all_labels = []  
    with torch.no_grad():   
        for inputs, labels,period in val_loader:  
            inputs = [input_.float().to(device) for input_ in inputs] 
            labels = labels.long()
            labels = labels.to(device) 
            
            outputs = model(inputs)  
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1) 

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())    
                
    all_preds = np.concatenate(all_preds)  
    all_labels = np.concatenate(all_labels) 
    
    auc = roc_auc_score(all_labels, all_preds) 
    torch.cuda.empty_cache()
    return auc, all_preds, all_labels

def get_contribution(model,train_loader):   # Get current predictions and true labels to compute single-modality contribution
    model = model.to(device)
    model.eval()    
    all_preds = []  
    all_labels = []  
    with torch.no_grad():   
        for inputs, labels,period in train_loader:
            inputs = [input_.float().to(device) for input_ in inputs] 
            labels = labels.long()
            labels = labels.to(device) 
            outputs = model(inputs) 
            preds = torch.sigmoid(outputs).squeeze()  
            all_preds.append(preds.cpu().numpy()[:, 1])  
            all_labels.append(labels.cpu().numpy())  
    all_preds = np.concatenate(all_preds)  
    all_labels = np.concatenate(all_labels)   
    torch.cuda.empty_cache()
    return all_labels, all_preds  


def get_dataset(combo):  
    data_frames = []   
    
    for element in combo: 
        data = ALL_data[int(element)]  
        data_frames.append(data) 

    return data_frames   

def get_dataset_TRAIN_data_subset(TRAIN_data_subset,combo,index):  # combo is an array where each element represents a filename from the files list
    data_frames = []  
    verify = []
    for element in combo: 
        data = TRAIN_data_subset[int(element)]  
        deleted_row = data[index]
        
        deleted_row = deleted_row.astype(float)

        verify.append(deleted_row)
        data = np.delete(data, index, axis=0)  
        data_frames.append(data) 
    
    return data_frames,verify
    
def get_contribution_value(true_labels,probs,combo):      #Calculates the contribution score based on correct negative predictions (label=0 and prob<0.5). 
    i = 0                                                 #Each correct negative prediction adds the length of the combination to the total score. 
    v = 0
    for label in true_labels:
       if(label == 0 and probs[i] <0.5):
            v += len(combo)
       i += 1
    return v

def get_contribution_TRAIN_data_subset(model,verify):   # Used to obtain current predictions and true labels for calculating single-modal contribution
    model.eval()      
    label = verify[0][0]
    verify_data = [row[2:] for row in verify]
    input_tensors = [torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device) for data in verify_data]
    model = model.to(device)
    
    with torch.no_grad():
        output = model(input_tensors)
    _, predicted_classes = torch.max(output, dim=1)
    
    return label,predicted_classes.cpu().numpy()
 
def Resampling_Level(d):
    if(d<0):
        return 3
    if(d==0):
        return 1
    if(d>0):
        return 2
    
     

def main():
    parser = argparse.ArgumentParser(description='Sample-level Evaluation Strategy')
    parser.add_argument('string1', type=str)
    parser.add_argument('string2', type=str)
    
    args = parser.parse_args()
    
    print(f"input path : {args.string1}")
    print(f"output path : {args.string2}")
    
    input_path = args.string1
    output_path = args.string2

    fix_all_random_seeds(42)

    train_model_num = 20 
    flag1=1
    #folder = "../dataset/10-fold-cross-validation/"
    folder = input_path

    for iii in range(5): 
        for ii in range(10):
            print(f"i={iii} ii={ii} (iii->5,ii->10) total=50")     
            Folder_Name = f"10-fold-cross-validation_{iii+1}_{ii}" 
            TRAIN_data = [None for _ in range(13)]  
            TEST_data = [None for _ in range(13)]
            ALL_data = [None for _ in range(13)]  
            ORIGINAL_Ytrain_data = [None for _ in range(13)] 
            ORIGINAL_Ytest_data = [None for _ in range(13)] 
            ORIGINAL_Ptrain_data = [None for _ in range(13)] 
            ORIGINAL_Ptest_data = [None for _ in range(13)] 
            ORIGINAL_IDtrain_data = [None for _ in range(13)] 
            ORIGINAL_IDtest_data = [None for _ in range(13)] 
            
            for pp in range(13):
                train_loader,test_loader,all_loader, original_y_train,original_y_test,original_p_train,original_p_test,original_id_train,original_id_test = creat_dataloader(folder,iii,ii,NAME[pp])
                ORIGINAL_Ytrain_data[pp] = original_y_train
                ORIGINAL_Ytest_data[pp] = original_y_test  
                ORIGINAL_Ptrain_data[pp] = original_p_train
                ORIGINAL_Ptest_data[pp] = original_p_test 
                TRAIN_data[pp] = train_loader
                TEST_data[pp] = test_loader
                ALL_data[pp] = all_loader
            
                ORIGINAL_IDtrain_data[pp] = original_id_train
                ORIGINAL_IDtest_data[pp] = original_id_test 
        
            TRAIN_data = [df.to_numpy() for df in TRAIN_data]
            TEST_data = [df.to_numpy() for df in TEST_data]
            ALL_data = [df.to_numpy() for df in ALL_data]
            
            #--------------------------------------------------------Model Pretraining  Warmup Epochs
            
            modal_input_sizes = [df.shape[1] - 2 for df in TRAIN_data]
            #modal_input_sizes2 = [size // 2 for size in modal_input_sizes]
            modal_input_sizes2 = [max(size // 2, 10) for size in modal_input_sizes]
    
            model = ELSM(modal_input_sizes, modal_input_sizes2 ,100 , 2)
                
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            dataset = CustomDataset(TRAIN_data) 
        
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(dataset, batch_size=30, shuffle=True)
            train_model(model, criterion, optimizer, train_loader, num_epochs = 20)   
            
            #--------------------------------------------------------Resampling-based Multimodal Fusion     
            
            dataset_test = CustomDataset(TEST_data)
            test_loader = DataLoader(dataset_test, batch_size=10, shuffle=True)  
            auc,PRE,LABEL = validate(test_loader , model)  
            print(f'Fold {ii+1} of {iii+1}th 10-fold CV - Test AUC: {auc:.4f}', flush=True)
            #----------------------------------------------------------------------------------------------
        
            total_numbers = range(0, TRAIN_data[0].shape[0]) 
            subset_size = int(len(total_numbers) * 0.1)  
            print(subset_size)
            random_subset = random.sample(total_numbers, subset_size)
            TRAIN_data_subset = [data[random_subset, :] for data in TRAIN_data] 
        
            numbers = list(range(13))
            all_combinations = []
            
            # Iterate to get all combinations of length 12  
            # Total of 13 modalities, we only consider combinations with exactly 12 elements
            for r in range(12, 13): 
                combinations = list(itertools.combinations(numbers, r))
                combinations = [tuple(map(str, comb)) for comb in combinations]
                all_combinations.extend(combinations)            
            permutations = all_combinations
        
            S=0
            tot=len(permutations)  
    
            selected_permutations = []
            for perm in permutations:
                if(S==tot):
                    break
                S+=1
                selected_permutations.append(perm)
        
            all_contribution_arrays = []
        
            for index in range(TRAIN_data_subset[0].shape[0]):
                # Iterate through all permutation possibilities in selected_permutations
                v2 =0
                v1 =0
                i = 0
                Contribution_array = [0] * 13    
                for L in range (0,13):   # Calculate each modality's contribution as (v2 - v1) difference
                    ssss=0
                    v2=0
                    v1=0
                    for perm in selected_permutations:
                        if( f'{L}' not in perm):
                            ssss+=1   # ssss: count of selected combinations
                            string_perm = [str(x) for x in perm]
                            data_t1,verify= get_dataset_TRAIN_data_subset(TRAIN_data_subset,string_perm,index)   # Remove the sample data of row 'index' to avoid self-prediction interference
        
                            dataset_t1 = CustomDataset(data_t1)
                            dataset_t1_loader = DataLoader(dataset_t1, batch_size=10, shuffle=True, drop_last=True)
                            
                            modal_input_sizes = [df.shape[1] - 2 for df in data_t1]
                            modal_input_sizes2 = [size // 2 for size in modal_input_sizes]
                            modal_input_sizes2 = [max(size // 2, 1) for size in modal_input_sizes]
                            model = ELSM(modal_input_sizes, modal_input_sizes2 ,100 , 2)
                                           
                            criterion = nn.CrossEntropyLoss()  
                            optimizer = optim.Adam(model.parameters(), lr=0.001)                   
                            train_model(model, criterion, optimizer, dataset_t1_loader, num_epochs=train_model_num)
                            true_labels, probs  = get_contribution_TRAIN_data_subset(model,verify)
                            if(true_labels==probs):
                                v1 += len(perm)       
                            else:
                                v1 += 0
                            
                            string_perm.append(f'{L}')   #Contribution value of modality L
                        
                            data_t2,verify= get_dataset_TRAIN_data_subset(TRAIN_data_subset,string_perm,index)
                            dataset_t2 = CustomDataset(data_t2)
                            dataset_t2_loader = DataLoader(dataset_t2, batch_size=10, shuffle=True, drop_last=True)
                            
                            modal_input_sizes = [df.shape[1] - 2 for df in data_t2]
                            modal_input_sizes2 = [size // 2 for size in modal_input_sizes]
                            modal_input_sizes2 = [max(size // 2, 1) for size in modal_input_sizes]
                            model = ELSM(modal_input_sizes, modal_input_sizes2 ,100 , 2)
                            
                            criterion = nn.CrossEntropyLoss()  
                            optimizer = optim.Adam(model.parameters(), lr=0.001)                   
                            train_model(model, criterion, optimizer, dataset_t2_loader, num_epochs=train_model_num)
                            true_labels, probs  = get_contribution_TRAIN_data_subset(model,verify)
                            if(true_labels==probs):
                                v2 += len(perm)+1        
                            else:
                                v2 += 0 
                                
                    Contribution= max(v2-v1,0)
                    
                    Contribution_array[L] += Contribution                          
                Contribution_array = [value / ssss for value in Contribution_array]
                all_contribution_arrays.append(Contribution_array)              
             
            total_greater_than_one=0
            total_smaller_than_one=0  
            position_lowCon = [0] * 13        
            for idx, contribution in enumerate(all_contribution_arrays):
                total_greater_than_one += sum(value for value in contribution if value > 1)
                total_smaller_than_one += sum(value for value in contribution if value <= 1)
                # Increment position_lowCon at indices where values are < 1
                for i, value in enumerate(contribution):
                    if value <= 1:  
                        position_lowCon[i] += 1 
                        
            d = (total_greater_than_one-total_smaller_than_one)/(TRAIN_data_subset[0].shape[0])
            d =  Resampling_Level(d)
            flag =0 
            for h in range(d):
                for m, value in enumerate(position_lowCon):
                    Mean_value = np.mean(position_lowCon)
                    
                   # Select low-contribution modalities 
                   # A modality is considered low-contribution if 50% of samples are identified as such
                    
                    if(value <= Mean_value): 
                        continue  
                    if(flag ==0):
                        a=TRAIN_data.copy()
                        b=ORIGINAL_IDtrain_data.copy()
                        c=ORIGINAL_Ytrain_data.copy()
                        d=ORIGINAL_Ptrain_data.copy()
                        flag=1
                    TRAIN_data[m] = np.concatenate([TRAIN_data[m], a[m]], axis=0)
                    ORIGINAL_IDtrain_data[m] = np.concatenate([ORIGINAL_IDtrain_data[m], b[m]], axis=0)
                    ORIGINAL_Ytrain_data[m] = np.concatenate([ORIGINAL_Ytrain_data[m], c[m]], axis=0)
                    ORIGINAL_Ptrain_data[m] = np.concatenate([ORIGINAL_Ptrain_data[m], d[m]], axis=0)
                    
            #directory_path = '../sample_level_evaluation_strategy_result/'
            directory_path = output_path
            folder_path = os.path.join(directory_path, Folder_Name)   
            os.makedirs(folder_path, exist_ok=True)
            
            directory_path = directory_path + Folder_Name
            
            # Store TRAIN_data and TEST_data
            
            D_train = [  
                'coverage_train',
                'delfi_train',  
                'end_train',  
                'end_motifs_train',  
                'fragment_train',  
                'fragmenter_train',  
                'FSD_train',  
                'FSR_train',  
                'IFS_train',
                'length_train',  
                'WPS_train',  
                'OCF_train',  
                'PFE_train'  
            ]
            
            D_test = [  
                'coverage_test',
                'delfi_test',  
                'end_test',  
                'end_motifs_test',  
                'fragment_test',  
                'fragmenter_test',  
                'FSD_test',  
                'FSR_test',  
                'IFS_test', 
                'length_test',  
                'WPS_test',  
                'OCF_test',  
                'PFE_test'  
            ]
            
     
            for i, filename in enumerate(D_train):  
                df = TRAIN_data[i]
                
                ORIGINAL_Ptrain_data[i] = pd.Series(ORIGINAL_Ptrain_data[i])
                ORIGINAL_Ptrain_data[i] = ORIGINAL_Ptrain_data[i].reset_index(drop=True)
            
                ORIGINAL_IDtrain_data[i] = pd.Series(ORIGINAL_IDtrain_data[i])
                ORIGINAL_IDtrain_data[i] = ORIGINAL_IDtrain_data[i].reset_index(drop=True)
                
                ORIGINAL_Ytrain_data[i] = pd.Series(ORIGINAL_Ytrain_data[i])
                ORIGINAL_Ytrain_data[i] = ORIGINAL_Ytrain_data[i].reset_index(drop=True)  
                df = pd.DataFrame(df) 
                df.iloc[:, 0] = ORIGINAL_Ytrain_data[i]
                df.iloc[:, 1] = ORIGINAL_Ptrain_data[i]  
                df.insert(0, 'ID', ORIGINAL_IDtrain_data[i].reset_index(drop=True))
                
                df = pd.DataFrame(df)  
                file_path = os.path.join(folder_path, f'{filename}.csv') 
                df.to_csv(file_path, index=False, compression='zip')
                
            
            for i, filename in enumerate(D_test):  
                df = TEST_data[i]
                df = pd.DataFrame(df)
                
                ORIGINAL_Ptest_data[i] = pd.Series(ORIGINAL_Ptest_data[i])
                ORIGINAL_Ptest_data[i] = ORIGINAL_Ptest_data[i].reset_index(drop=True)
            
                ORIGINAL_IDtest_data[i] = pd.Series(ORIGINAL_IDtest_data[i])
                ORIGINAL_IDtest_data[i] = ORIGINAL_IDtest_data[i].reset_index(drop=True)
                
                ORIGINAL_Ytest_data[i] = pd.Series(ORIGINAL_Ytest_data[i])
                ORIGINAL_Ytest_data[i] = ORIGINAL_Ytest_data[i].reset_index(drop=True) 
                df.iloc[:, 0] = ORIGINAL_Ytest_data[i] 
                df.iloc[:, 1] = ORIGINAL_Ptest_data[i]
                df.insert(0, 'ID', ORIGINAL_IDtest_data[i].reset_index(drop=True))
                
                df = pd.DataFrame(df)  
                file_path = os.path.join(folder_path, f'{filename}.csv') 
                df.to_csv(file_path, index=False, compression='zip')    
        
if __name__ == "__main__":
    main()

#python3 sample_level_evaluation_strategy.py "../dataset/10-fold-cross-validation/" "../sample_level_evaluation_strategy_result/"


'''
nohup python3 sample_level_evaluation_strategy_cross.py \
"../dataset/10-fold-cross-validation/" \
"../sample_level_evaluation_strategy_result/" \
> sample_level_evaluation_strategy_cross.out 2>&1 &
'''


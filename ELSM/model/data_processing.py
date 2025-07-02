# coding=gb2312 
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

roman_to_arabic = {
     0: 0,
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4,
    'V': 5,
    'X': 6
}

def get_val_datasetlist(DIR):
    train_data_coverage = pd.read_csv(f'{DIR}coverage_features_train.csv', compression='zip')
    train_data_delfi = pd.read_csv(f'{DIR}delfi_features_train.csv', compression='zip')
    train_data_endf = pd.read_csv(f'{DIR}end_features_train.csv', compression='zip')
    train_data_end = pd.read_csv(f'{DIR}end_motifs_features_train.csv', compression='zip')
    train_data_fragment = pd.read_csv(f'{DIR}fragment_features_train.csv', compression='zip')
    train_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_features_train.csv', compression='zip')
    train_data_fsd = pd.read_csv(f'{DIR}FSD_features_train.csv', compression='zip')
    train_data_fsr = pd.read_csv(f'{DIR}FSR_features_train.csv', compression='zip')
    train_data_length = pd.read_csv(f'{DIR}length_features_train.csv', compression='zip')  
    train_data_WPS = pd.read_csv(f'{DIR}WPS_features_train.csv', compression='zip')  
    train_data_OCF = pd.read_csv(f'{DIR}OCF_features_train.csv', compression='zip')
    train_data_PFE = pd.read_csv(f'{DIR}PFE_features_train.csv', compression='zip')
    train_data_ifs = pd.read_csv(f'{DIR}IFS_features_train.csv', compression='zip')
    
    train_df_list = [
        train_data_coverage,
        train_data_delfi,
        train_data_endf,
        train_data_end,
        train_data_fragment,
        train_data_fragmenter,
        train_data_fsd,
        train_data_fsr,
        train_data_ifs,
        train_data_length,
        train_data_WPS,
        train_data_OCF,
        train_data_PFE
    ]
    
    test_data_coverage = pd.read_csv(f'{DIR}coverage_features_valid.csv', compression='zip')
    test_data_delfi = pd.read_csv(f'{DIR}delfi_features_valid.csv', compression='zip')
    test_data_endf = pd.read_csv(f'{DIR}end_features_valid.csv', compression='zip')
    test_data_end = pd.read_csv(f'{DIR}end_motifs_features_valid.csv', compression='zip')
    test_data_fragment = pd.read_csv(f'{DIR}fragment_features_valid.csv', compression='zip')
    test_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_features_valid.csv', compression='zip')
    test_data_fsd = pd.read_csv(f'{DIR}FSD_features_valid.csv', compression='zip')
    test_data_fsr = pd.read_csv(f'{DIR}FSR_features_valid.csv', compression='zip')
    test_data_length = pd.read_csv(f'{DIR}length_features_valid.csv', compression='zip')  
    test_data_WPS = pd.read_csv(f'{DIR}WPS_features_valid.csv', compression='zip')  
    test_data_OCF = pd.read_csv(f'{DIR}OCF_features_valid.csv', compression='zip')
    test_data_PFE = pd.read_csv(f'{DIR}PFE_features_valid.csv', compression='zip')
    test_data_ifs = pd.read_csv(f'{DIR}IFS_features_valid.csv', compression='zip')
    
    test_df_list = [
        test_data_coverage,
        test_data_delfi,
        test_data_endf,
        test_data_end,
        test_data_fragment,
        test_data_fragmenter,
        test_data_fsd,
        test_data_fsr,
        test_data_ifs,
        test_data_length,
        test_data_WPS,
        test_data_OCF,
        test_data_PFE
    ]
    return train_df_list,test_df_list


def get_val_datasetlist_execution(DIR):
    train_data_coverage = pd.read_csv(f'{DIR}coverage_train.csv', compression='zip')
    train_data_delfi = pd.read_csv(f'{DIR}delfi_train.csv', compression='zip')
    train_data_endf = pd.read_csv(f'{DIR}end_train.csv', compression='zip')
    train_data_end = pd.read_csv(f'{DIR}end_motifs_train.csv', compression='zip')
    train_data_fragment = pd.read_csv(f'{DIR}fragment_train.csv', compression='zip')
    train_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_train.csv', compression='zip')
    train_data_fsd = pd.read_csv(f'{DIR}FSD_train.csv', compression='zip')
    train_data_fsr = pd.read_csv(f'{DIR}FSR_train.csv', compression='zip')
    train_data_length = pd.read_csv(f'{DIR}length_train.csv', compression='zip')  
    train_data_WPS = pd.read_csv(f'{DIR}WPS_train.csv', compression='zip')  
    train_data_OCF = pd.read_csv(f'{DIR}OCF_train.csv', compression='zip')
    train_data_PFE = pd.read_csv(f'{DIR}PFE_train.csv', compression='zip')
    train_data_ifs = pd.read_csv(f'{DIR}IFS_train.csv', compression='zip')
    
    train_df_list = [
        train_data_coverage,
        train_data_delfi,
        train_data_endf,
        train_data_end,
        train_data_fragment,
        train_data_fragmenter,
        train_data_fsd,
        train_data_fsr,
        train_data_ifs,
        train_data_length,
        train_data_WPS,
        train_data_OCF,
        train_data_PFE
    ]
    
    test_data_coverage = pd.read_csv(f'{DIR}coverage_test.csv', compression='zip')
    test_data_delfi = pd.read_csv(f'{DIR}delfi_test.csv', compression='zip')
    test_data_endf = pd.read_csv(f'{DIR}end_test.csv', compression='zip')
    test_data_end = pd.read_csv(f'{DIR}end_motifs_test.csv', compression='zip')
    test_data_fragment = pd.read_csv(f'{DIR}fragment_test.csv', compression='zip')
    test_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_test.csv', compression='zip')
    test_data_fsd = pd.read_csv(f'{DIR}FSD_test.csv', compression='zip')
    test_data_fsr = pd.read_csv(f'{DIR}FSR_test.csv', compression='zip')
    test_data_length = pd.read_csv(f'{DIR}length_test.csv', compression='zip')  
    test_data_WPS = pd.read_csv(f'{DIR}WPS_test.csv', compression='zip')  
    test_data_OCF = pd.read_csv(f'{DIR}OCF_test.csv', compression='zip')
    test_data_PFE = pd.read_csv(f'{DIR}PFE_test.csv', compression='zip')
    test_data_ifs = pd.read_csv(f'{DIR}IFS_test.csv', compression='zip')
    
    test_df_list = [
        test_data_coverage,
        test_data_delfi,
        test_data_endf,
        test_data_end,
        test_data_fragment,
        test_data_fragmenter,
        test_data_fsd,
        test_data_fsr,
        test_data_ifs,
        test_data_length,
        test_data_WPS,
        test_data_OCF,
        test_data_PFE
    ]
    return train_df_list,test_df_list



class CustomDataset(Dataset):  
    def __init__(self, data_list):    
    
        self.label_encoder = LabelEncoder()
        all_labels = np.concatenate([data[:, 0] for data in data_list])
        self.label_encoder.fit(all_labels)
        
        self.data_list = data_list
        
           
        self.labels = [data[:, 0] for data in self.data_list]  
        
        self.features = [data[:, 2:] for data in self.data_list]  
        
        self.period = [data[:, 1] for data in self.data_list]  
        
        assert all(len(labels) == len(features) for labels, features in zip(self.labels, self.features))  
        
        self.num_samples = len(self.labels[0])  
  
    def __len__(self):  
        return self.num_samples  
  
    def __getitem__(self, idx):  

        features = [torch.tensor(features[idx, :], dtype=torch.float32) for features in self.features]  

        labels = torch.tensor(self.labels[0][idx], dtype=torch.float32)  
        
        period = torch.tensor(self.period[0][idx], dtype=torch.float32)
        return features, labels ,period 
    def get_classes(self):
        return {original: encoded for encoded, original in enumerate(self.label_encoder.classes_)}
        

def filter_files(file, name, ii):
    file_parts = file.split('.')[0].split('_')

    if name == "end":
        if file_parts[0] == 'end' and file_parts[1] == 'features' and file_parts[-2] == f'{ii}':
            return True
    elif name == "end_motifs":
        if file_parts[0] == 'end' and file_parts[1] == 'motifs' and file_parts[-2] == f'{ii}':
            return True
    else:
        if file_parts[0] == name and file_parts[-2] == f'{ii}':
            return True
    return False



def creat_dataloader_val(filedata1,filedata2):
    
    filedata1.iloc[:, 2] = filedata1.iloc[:, 2].map(roman_to_arabic)
    filedata2.iloc[:, 2] = filedata2.iloc[:, 2].map(roman_to_arabic)

    filedata_combined = pd.concat([filedata1, filedata2], axis=0, ignore_index=True)

    filedata_combined = filedata_combined.drop(filedata_combined.columns[[0, 2]], axis=1)
  
    all_data_with_label=filedata_combined
    
     
    X_train = filedata1.iloc[:, [0] + list(range(2, filedata1.shape[1]))].values 
    y_train = filedata1.iloc[:, 1].values  
    X_test = filedata2.iloc[:, [0] + list(range(2, filedata2.shape[1]))].values 
    y_test = filedata2.iloc[:, 1].values   
    
    original_id_train = X_train[:, 0]
    original_id_test = X_test[:, 0]
    X_test = X_test[:, 1:]
    X_train = X_train[:, 1:]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    original_y_train = y_train
    original_y_test = y_test    
    original_p_train = X_train.iloc[:, 0].copy()
    original_p_test = X_test.iloc[:, 0].copy()
    
    y_train = np.where(np.isin(y_train, ['healthy', 'Benign', 'No baseline cancer']), 0, 1)
    y_test = np.where(np.isin(y_test, ['healthy', 'Benign', 'No baseline cancer']), 0, 1)         
    
    X_train.insert(0, 'label', y_train)  
    X_test.insert(0, 'label', y_test) 

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
   
    return X_train ,X_test ,all_data_with_label, original_y_train, original_y_test,  original_p_train,  original_p_test, original_id_train, original_id_test

 

def creat_dataloader(folder, i,ii,name):
    path = f"{folder}{i+1}/"
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    file_parts = [
        file for file in csv_files
        if filter_files(file, name, ii)
    ]
    name_train=''
    name_test=''   
    for file in file_parts:
        name = os.path.splitext(os.path.basename(file))[0].rstrip('.csv')
        if(name.split('_')[-1]=='train'):
            name_train = file
        else:
            name_test = file
    
    train = pd.read_csv(f'{path}{name_train}', compression='gzip')
    test = pd.read_csv(f'{path}{name_test}', compression='gzip')
    
    combined_data = pd.concat([train, test], axis=0, ignore_index=True)
    X = combined_data.iloc[:, 3:]
    y = combined_data.iloc[:, 1]
    
    train.loc[train.iloc[:, 1].isin(['healthy', 'Benign', 'No baseline cancer']), train.columns[2]] = 0    
    train.iloc[:, 2] = train.iloc[:, 2].map(roman_to_arabic)
    test.loc[test.iloc[:, 1].isin(['healthy', 'Benign', 'No baseline cancer']), test.columns[2]] = 0
    test.iloc[:, 2] = test.iloc[:, 2].map(roman_to_arabic)    
    
    X_train = train.iloc[:, [0] + list(range(2, train.shape[1]))].values
    y_train = train.iloc[:, 1].values
    X_test = test.iloc[:, [0] + list(range(2, test.shape[1]))].values
    y_test = test.iloc[:, 1].values

    original_id_train = X_train[:, 0]
    original_id_test = X_test[:, 0]
    X_test = X_test[:, 1:]
    X_train = X_train[:, 1:]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    original_y_train = y_train
    original_y_test = y_test    
    original_p_train = X_train.iloc[:, 0].copy()
    original_p_test = X_test.iloc[:, 0].copy()
    
    y_train = np.where(np.isin(y_train, ['healthy', 'Benign', 'No baseline cancer']), 0, 1)
    y_test = np.where(np.isin(y_test, ['healthy', 'Benign', 'No baseline cancer']), 0, 1)         
    X_train.insert(0, 'label', y_train)  
    X_test.insert(0, 'label', y_test) 
    
    train_data =  X_train
    test_data = X_test
        
    all_data_with_label = X.copy()  
    all_data_with_label = pd.DataFrame(all_data_with_label)
    all_data_with_label.insert(0, 'label', y)
    
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    
    return train_data ,test_data ,all_data_with_label, original_y_train, original_y_test,  original_p_train,  original_p_test, original_id_train, original_id_test



def creat_dataloader_exe(filedata1,filedata2,Random_State):   
    X_train = filedata1.iloc[:, 2:].values
    y_train = filedata1.iloc[:, 1].values 
    
    X_test = filedata2.iloc[:, 2:].values
    y_test = filedata2.iloc[:, 1].values 

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    
    y_test = pd.Series(y_test)
    y_test = np.where(y_test.isin(['healthy', 'Benign', 'No baseline cancer']), 0, 1)
    y_train = pd.Series(y_train)
    y_train = np.where(y_train.isin(['healthy', 'Benign', 'No baseline cancer']), 0, 1)
          
    X_train.insert(0, 'label', y_train)  
    X_test.insert(0, 'label', y_test) 
    
    train_data =  X_train
    test_data = X_test
 
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    all_data_with_label = X.copy()   
    all_data_with_label = pd.DataFrame(all_data_with_label)
    all_data_with_label.insert(0, 'label', y) 
    
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    return train_data ,test_data,all_data_with_label



def read_train_csv_execross(dirname,path):
    DIR = f"{path}{dirname}/"
    train_data_coverage = pd.read_csv(f'{DIR}coverage_train.csv', compression='zip')
    train_data_delfi = pd.read_csv(f'{DIR}delfi_train.csv', compression='zip')
    train_data_endf = pd.read_csv(f'{DIR}end_train.csv', compression='zip')
    train_data_end = pd.read_csv(f'{DIR}end_motifs_train.csv', compression='zip')
    train_data_fragment = pd.read_csv(f'{DIR}fragment_train.csv', compression='zip')
    train_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_train.csv', compression='zip')
    train_data_fsd = pd.read_csv(f'{DIR}FSD_train.csv', compression='zip')
    train_data_fsr = pd.read_csv(f'{DIR}FSR_train.csv', compression='zip')
    train_data_length = pd.read_csv(f'{DIR}length_train.csv', compression='zip')  
    train_data_WPS = pd.read_csv(f'{DIR}WPS_train.csv', compression='zip')  
    train_data_OCF = pd.read_csv(f'{DIR}OCF_train.csv', compression='zip')
    train_data_PFE = pd.read_csv(f'{DIR}PFE_train.csv', compression='zip')
    train_data_ifs = pd.read_csv(f'{DIR}IFS_train.csv', compression='zip')
    
    train_df_list = [
        train_data_coverage,
        train_data_delfi,
        train_data_endf,
        train_data_end,
        train_data_fragment,
        train_data_fragmenter,
        train_data_fsd,
        train_data_fsr,
        train_data_ifs,
        train_data_length,
        train_data_WPS,
        train_data_OCF,
        train_data_PFE
    ]
    return train_df_list

def read_test_csv_execross(dirname,path):
    DIR = f"{path}{dirname}/"
    test_data_coverage = pd.read_csv(f'{DIR}coverage_test.csv', compression='zip')
    test_data_delfi = pd.read_csv(f'{DIR}delfi_test.csv', compression='zip')
    test_data_endf = pd.read_csv(f'{DIR}end_test.csv', compression='zip')
    test_data_end = pd.read_csv(f'{DIR}end_motifs_test.csv', compression='zip')
    test_data_fragment = pd.read_csv(f'{DIR}fragment_test.csv', compression='zip')
    test_data_fragmenter = pd.read_csv(f'{DIR}fragmenter_test.csv', compression='zip')
    test_data_fsd = pd.read_csv(f'{DIR}FSD_test.csv', compression='zip')
    test_data_fsr = pd.read_csv(f'{DIR}FSR_test.csv', compression='zip')
    test_data_length = pd.read_csv(f'{DIR}length_test.csv', compression='zip')  
    test_data_WPS = pd.read_csv(f'{DIR}WPS_test.csv', compression='zip')  
    test_data_OCF = pd.read_csv(f'{DIR}OCF_test.csv', compression='zip')
    test_data_PFE = pd.read_csv(f'{DIR}PFE_test.csv', compression='zip')
    test_data_ifs = pd.read_csv(f'{DIR}IFS_test.csv', compression='zip')
    
    test_df_list = [
        test_data_coverage,
        test_data_delfi,
        test_data_endf,
        test_data_end,
        test_data_fragment,
        test_data_fragmenter,
        test_data_fsd,
        test_data_fsr,
        test_data_ifs,
        test_data_length,
        test_data_WPS,
        test_data_OCF,
        test_data_PFE
    ]
    return test_df_list


def creat_dataloader_execross(filedata1,filedata2):
    filedata1 = filedata1.dropna(subset=[filedata1.columns[1]])

    X = filedata1.iloc[:, 3:]  
    y = filedata1.iloc[:, 1]   
    y = y.replace({
        "healthy": 3,
        "No baseline cancer": 3,
        "Benign": 2
    })  
    y = y.apply(lambda x: 1 if x not in [3, 2] else x)
    
    X = pd.DataFrame(X)  
    X.insert(0, 'label', y)     
    train_data = X
    
    filedata2 = filedata2.dropna(subset=[filedata2.columns[1]])

    X = filedata2.iloc[:, 3:]  
    y = filedata2.iloc[:, 1]      
    y = y.replace({
        "healthy": 3,
        "No baseline cancer": 3,
        "Benign": 2
    })
    y = y.apply(lambda x: 1 if x not in [3, 2] else x)
    
            
    X = pd.DataFrame(X)  
    X.insert(0, 'label', y)     
    test_data = X    

    all_data_with_label = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    
    return train_data ,test_data ,all_data_with_label
    
    
class CustomDataset_execross(Dataset):  
    def __init__(self, data_list):   
        self.data_list = data_list    
        self.labels = [data[:, 0] for data in data_list]  
        self.features = [data[:, 1:] for data in data_list]  
        assert all(len(labels) == len(features) for labels, features in zip(self.labels, self.features))  
        self.num_samples = len(self.labels[0])  
    def __len__(self):  
        return self.num_samples  
    def __getitem__(self, idx):  
        features = []
        for feature in self.features:
            if idx >= len(feature):
                idx = idx % len(feature) 
            features.append(torch.tensor(feature[idx, :], dtype=torch.float32))   
        labels = self.labels[0][idx]  
        
        return features, labels  



    
    

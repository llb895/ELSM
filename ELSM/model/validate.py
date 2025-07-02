# coding=gb2312
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_val(val_loader, model):
    model = model.to(device)
    model.eval()    
    all_dfs_1 = []
    all_dfs_1and2 = []
    
    with torch.no_grad():  
        for inputs, labels,period in val_loader:  
            inputs = [input_.float().to(device) for input_ in inputs] 
            labels = labels.long().to(device)
            period = period.to(device)
            
            outputs = model(inputs)  
            probs = torch.softmax(outputs, dim=1) 
            
            preds = torch.argmax(probs, dim=1)
            preds = probs[:, 1]
            
            matrix = torch.cat((labels.view(-1, 1), period.view(-1, 1), preds.view(-1, 1)), dim=1)
            df = pd.DataFrame(matrix.cpu().numpy(), columns=["original_labels", "period", "probs"])
            all_dfs_1.append(df) 
  
    final_df_1 = pd.concat(all_dfs_1, axis=0, ignore_index=True)    
    torch.cuda.empty_cache()
    return final_df_1
    
    
def validate_cross(val_loader, model):  
    model = model.to(device)
    model.eval()   
    data = []
    all_labels = []  
    all_probs = []  
    with torch.no_grad():  
        for inputs, labels in val_loader:  
            inputs = [input_.float().to(device) for input_ in inputs] 
            labels = labels.to(device)
            original_labels = labels.clone()
            
            labels[labels == 2] = 0
            labels[labels == 3] = 0  
            
            outputs = model(inputs)  
            probabilities = F.softmax(outputs, dim=1)
            
            selected_probs = probabilities[:, 1]
            data.extend(list(zip(original_labels.cpu().numpy(), selected_probs.cpu().numpy())))
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(selected_probs.cpu().numpy())             
                    
    big_df = pd.DataFrame(data, columns=['labels', 'score'])
    auc = roc_auc_score(all_labels, all_probs)
    torch.cuda.empty_cache()
    return big_df,auc
    
    
    
# coding=gb2312 

import torch  
from torch.utils.data import DataLoader, Subset  
import torch.nn as nn  
import torch.nn.functional as F  
  
class SingleModalNet(nn.Module):  
    def __init__(self, input_size,input_size2, hidden_size, num_classes):  
        super(SingleModalNet, self).__init__()  
        self.fc0 = nn.Linear(input_size, input_size2)
        self.bn0 = nn.BatchNorm1d(input_size2)
        self.fc1 = nn.Linear(input_size2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):  
        out = self.fc0(x) 
        out = self.bn0(out)
        out = self.fc1(out)  
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)  
        return out 

'''
The model first processes each modality's input with a set of sub-networks to obtain preliminary features. 
It then concatenates the raw input with these initial features and uses another set of sub-networks to further extract features. 
Finally, features from all modalities are concatenated and fused through a fully-connected layer to produce the final output.
'''
class ELSM(nn.Module):
    def __init__(self, modal_input_sizes, modal_input_sizes2, hidden_size, output_size, modal_sample_counts):
        super(ELSM, self).__init__()
        self.modal_sample_counts = modal_sample_counts  
        self.modal_weights = self.compute_modal_weights(modal_sample_counts)  
        self.networks = nn.ModuleList([
            SingleModalNet(input_size, input_size2, hidden_size, hidden_size)
            for input_size, input_size2 in zip(modal_input_sizes, modal_input_sizes2)
        ])
        self.networks1 = nn.ModuleList([
            SingleModalNet(input_size + hidden_size, input_size2 + hidden_size, hidden_size, hidden_size)
            for input_size, input_size2 in zip(modal_input_sizes, modal_input_sizes2)
        ])   
        self.fusion = nn.Linear(hidden_size * len(modal_input_sizes), output_size)  
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.tensor(self.compute_modal_weights(modal_sample_counts), dtype=torch.float32))
    
    def compute_modal_weights(self, modal_sample_counts):
        total_samples = sum(modal_sample_counts)
        modal_weights = [count / total_samples for count in modal_sample_counts]
        return modal_weights

    def forward(self, inputs):
        for i in range(2):
            if(i == 0):
                outputs = [net(input_) for net, input_ in zip(self.networks, inputs)]
                weights = torch.ones(len(outputs)) 
                weights = weights / weights.sum()
                fused_output = sum(weight * output for weight, output in zip(weights, outputs))                
                context_t = fused_output

            if(i == 1):
                concatenated_inputs = []
                for i in range(len(inputs)):
                    input_tensor = inputs[i]
                    context_tensor = context_t
                    concatenated = torch.cat((input_tensor, context_tensor), dim=-1)
                    concatenated_inputs.append(concatenated)
                inputs = concatenated_inputs
                outputs = [net(input_) for net, input_ in zip(self.networks1, inputs)]
                outputs = torch.stack(outputs, dim=0)
                context_t =  outputs
        weighted_outputs = []
        for i in range(outputs.size(0)): 
            weighted_output = outputs[i] * self.attention_weights[i]  
            weighted_outputs.append(weighted_output)
        weighted_outputs = torch.stack(weighted_outputs, dim=0)
        outputs = weighted_outputs
                         
        outputs = outputs.permute(1, 0, 2)
        concatenated = outputs.flatten(start_dim=1)
        fused = self.fusion(concatenated) 
        
        return fused
        

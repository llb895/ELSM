# coding=gb2312
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
def train_model_val(model, criterion, optimizer,train_loader, num_epochs): 
    model = model.to(device)
    for epoch in range(num_epochs): 
        model.train() 
        total_loss = 0   
        for inputs, labels, period in train_loader:  
         
            inputs = [input_.float().to(device) for input_ in inputs] 
            
            
            labels = labels.long().to(device)
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


def train_model_cross(model, criterion, optimizer,train_loader, num_epochs):   
    model.to(device)
    for epoch in range(num_epochs): 
        total_loss = 0   
        for inputs, labels in train_loader:   
            inputs = [input_.float().to(device) for input_ in inputs]
            labels = labels.to(device)
            
            labels[labels == 2] = 0
            labels[labels == 3] = 0  
            labels = labels.long()

            optimizer.zero_grad()   
            outputs = model(inputs)  
            loss = criterion(outputs, labels)   
            loss.backward()  
            optimizer.step()
            total_loss += loss.item() 
        avg_loss = total_loss / len(train_loader)  
    torch.cuda.empty_cache()
        
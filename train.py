import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    if scheduler is not None: 
        lr_list = []
        lr_list.append(optimizer.param_groups[0]['lr'])
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        if device.type == 'mps': # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
        else:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        if scaler:
            if device.type == 'cuda':
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif device.type == 'mps':
                with autocast('mps'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # cpu, defensive coding
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        else: #No mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'batch': batch_idx+1,
            'loss': running_loss/len(train_loader),
            'accuracy': 100.*correct/total,
            'lr': optimizer.param_groups[0]['lr']   
        })
    train_state_dict = {
        'name': 'train',
        'loss': running_loss/len(train_loader),
        'accuracy': 100.*correct/total
    }
    if scheduler is not None:
        lr_list = [optimizer.param_groups[0]['lr']]
        train_state_dict['lr_list'] = lr_list
        train_state_dict['scheduler'] = scheduler.state_dict()

    return train_state_dict
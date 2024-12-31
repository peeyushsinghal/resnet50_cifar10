import torch
from tqdm import tqdm
from torch.amp import autocast

def test(model, test_loader, criterion, device, mix_precision=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            if mix_precision:
                if device.type == 'cuda':
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                elif device.type == 'mps':
                    with autocast('mps'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:  # cpu, defensive coding
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/len(test_loader),
                'accuracy': 100.*correct/total
            })

    test_state_dict = {
        'name': 'test',
        'loss': running_loss/len(test_loader),
        'accuracy': 100.*correct/total
    }

    return test_state_dict    
    # return running_loss/len(test_loader), 100.*correct/total
import torch
from torchvision.datasets import CIFAR10
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import yaml

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label

def get_transforms(config, train=True):
    if train:
        transform = A.Compose([
            A.Resize(height=config['transforms']['resize']['height'],
                    width=config['transforms']['resize']['width']),
            A.CenterCrop(height=config['transforms']['center_crop']['height'],
                        width=config['transforms']['center_crop']['width']),
            A.HorizontalFlip(p=config['transforms']['horizontal_flip_prob']),
            A.ShiftScaleRotate(shift_limit=config['transforms']['shift_scale_rotate']['shift_limit'],
                             scale_limit=config['transforms']['shift_scale_rotate']['scale_limit'],
                             rotate_limit=config['transforms']['shift_scale_rotate']['rotate_limit'],
                             p=config['transforms']['shift_scale_rotate']['p']),
            A.CoarseDropout(max_holes=config['transforms']['coarse_dropout']['max_holes'],
                          max_height=config['transforms']['coarse_dropout']['max_height'],
                          max_width=config['transforms']['coarse_dropout']['max_width'],
                          p=config['transforms']['coarse_dropout']['p']),
            A.Normalize(mean=config['data']['mean'], std=config['data']['std']),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(height=config['transforms']['resize']['height'],
                    width=config['transforms']['resize']['width']),
            A.Normalize(mean=config['data']['mean'], std=config['data']['std']),
            ToTensorV2()
        ])
    return transform

def get_dataloaders(config):
    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)
    
    train_dataset = CIFAR10Dataset(train=True, transform=train_transform)
    test_dataset = CIFAR10Dataset(train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, test_loader 
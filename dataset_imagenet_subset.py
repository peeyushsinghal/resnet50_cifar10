import os
import torch
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Path to ILSVRC directory containing Data/CLS-LOC
            split (string): 'train' or 'val'
            transform: Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, 'Data', 'CLS-LOC', split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label

def get_transforms(config, train=True):
    if train:
        transform = A.Compose([
            A.RandomResizedCrop(
                height=config['transforms']['resize']['height'],
                width=config['transforms']['resize']['width']
            ),
            A.HorizontalFlip(p=config['transforms']['horizontal_flip_prob']),
            A.ShiftScaleRotate(
                shift_limit=config['transforms']['shift_scale_rotate']['shift_limit'],
                scale_limit=config['transforms']['shift_scale_rotate']['scale_limit'],
                rotate_limit=config['transforms']['shift_scale_rotate']['rotate_limit'],
                p=config['transforms']['shift_scale_rotate']['p']
            ),
            A.CoarseDropout(
                max_holes=config['transforms']['coarse_dropout']['max_holes'],
                max_height=config['transforms']['coarse_dropout']['max_height'],
                max_width=config['transforms']['coarse_dropout']['max_width'],
                p=config['transforms']['coarse_dropout']['p']
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(
                height=config['transforms']['resize']['height'],
                width=config['transforms']['resize']['width']
            ),
            A.CenterCrop(
                height=config['transforms']['center_crop']['height'],
                width=config['transforms']['center_crop']['width']
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ])
    return transform

def get_dataloaders(config):
    """
    Create train and validation dataloaders for ImageNet subset
    """
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    
    # Create datasets
    train_dataset = ImageNetSubsetDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNetSubsetDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        transform=val_transform
    )
    
    # Update num_classes in config based on actual classes in dataset
    config['model']['num_classes'] = len(train_dataset.classes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader 
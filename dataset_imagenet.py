import os
import torch
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with validation images and val_annotations.txt
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'ILSVRC2012_img_val')
        self.ann_file = os.path.join(root_dir, 'val_annotations.txt')
        
        # Read annotations
        self.val_data = []
        with open(self.ann_file, 'r') as f:
            for line in f:
                img_name, class_id = line.strip().split()[:2]
                self.val_data.append((img_name, int(class_id)))

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, idx):
        img_name, label = self.val_data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
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

class AlbumentationsImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, target

def get_imagenet_dataloaders(config):
    """
    Create train and validation dataloaders for ImageNet
    
    Directory structure should be:
    data_root/
        train/
            class1/
                img1.jpeg
                ...
            class2/
                img2.jpeg
                ...
        val/
            ILSVRC2012_img_val/
                ILSVRC2012_val_00000001.JPEG
                ...
            val_annotations.txt
    """
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    
    # Training dataset using ImageFolder
    train_dataset = AlbumentationsImageFolder(
        root=os.path.join(config['data']['root_dir'], 'train'),
        transform=train_transform
    )
    
    # Validation dataset using custom dataset
    val_dataset = ImageNetValidationDataset(
        root_dir=os.path.join(config['data']['root_dir'], 'val'),
        transform=val_transform
    )
    
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
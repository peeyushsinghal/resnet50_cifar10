import torch
import yaml
import math
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
import copy

from model import ResNet50
from dataset import get_dataloaders
from train import train_epoch
from test import test
from torchsummary import summary


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model_summary(model, input_size=(3, 32, 32)):
    summary(model, input_size=input_size)   

def find_lr(model, train_loader, config, device, criterion=None, start_lr=1e-7, end_lr=10, num_iter=100):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    _, suggested_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    
    return suggested_lr

def main():
    config = load_config()
    device = get_device()
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Initialize model
    model = ResNet50().to(device)
    get_model_summary(model)
    model_lr = copy.deepcopy(model)

    # Find the optimal learning rate
    print("Finding optimal learning rate...")
    suggested_lr = find_lr(model_lr, train_loader, config, device, criterion=None, start_lr=1e-7, end_lr=10, num_iter=100)
    config['training']['scheduler']['max_lr'] = suggested_lr
    print(f"Suggested maximum learning rate: {suggested_lr}")
    
    # # Define optimizer
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config['training']['scheduler']['max_lr']/config['training']['scheduler']['div_factor'],
    #     momentum=config['training']['optimizer']['momentum'],
    #     weight_decay=config['training']['optimizer']['weight_decay']
    # )
    
    # # Calculate total steps for OneCycleLR
    # total_steps = len(train_loader) * config['training']['epochs']
    
    # # Define scheduler
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=config['training']['scheduler']['max_lr'],
    #     total_steps=total_steps,
    #     pct_start=config['training']['scheduler']['pct_start'],
    #     div_factor=config['training']['scheduler']['div_factor'],
    #     final_div_factor=config['training']['scheduler']['final_div_factor']
    # )
    
    # criterion = torch.nn.CrossEntropyLoss()
    
    # # Training loop
    # for epoch in range(config['training']['epochs']):
    #     print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
    #     train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
    #     test_loss, test_acc = test(model, test_loader, criterion, device)
        
    #     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    #     print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main() 
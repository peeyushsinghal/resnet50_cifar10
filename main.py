import torch
import yaml
import math
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
import copy
from torch.amp import GradScaler

from model import ResNet50
# from dataset import get_dataloaders
from dataset_imagenet_subset import get_dataloaders
from train import train_epoch
from test import test
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")
import os
import json

class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
        
    def log_metrics(self, epoch_metrics):
        self.metrics.append(epoch_metrics)
        with open(os.path.join(self.log_dir, 'training_log.json'), 'a') as f:
            json.dump(self.metrics, f, indent=4)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path="config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model_summary(model, input_size=(3, 32, 32)):
    summary(model.to(torch.device("cpu")), input_size=input_size, device="cpu")


def find_lr(
    model,
    train_loader,
    config,
    device,
    criterion=None,
    start_lr=1e-7,
    end_lr=10,
    num_iter=100,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["optimizer"]["lr"],
        momentum=config["training"]["optimizer"]["momentum"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
    )
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter
    )
    _, suggested_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state

    return suggested_lr


def main():
    torch.cuda.empty_cache()
    config = load_config(config_path = 'config_imagenet_subset.yml')
    
    device = get_device()

    # Get data loaders
    print(f"-----Dataset {config['data']['dataset']} is being used.")
    train_loader, test_loader = get_dataloaders(config)
    print(f"-----Data loaders for {config['data']['dataset']} are loaded.")

    # Initialize model
    if device.type == "mps":
        model = ResNet50(num_classes=config['model']['num_classes']).to(device, dtype=torch.float32)
    else:
        model = ResNet50(num_classes=config['model']['num_classes']).to(device)

    if config["model"]["print_summary"]:
        get_model_summary(model)
    else:
        print("Model summary printining is supressed.")

    if config["model"]["print_model"]:
        print(model)
    else:
        print("Model printing is supressed.")

    model_lr = copy.deepcopy(model)

    # Find the optimal learning rate
    if config["training"]["lr_finder"]["enabled"]:
        print("-----Finding optimal learning rate...No training is done.")
        suggested_lr = find_lr(
            model_lr,
            train_loader,
            config,
            device,
            criterion=None,
            start_lr=float(config['training']['lr_finder']['start_lr']),
            end_lr=float(config['training']['lr_finder']['end_lr']),
            num_iter=100,
        )
        config["training"]["scheduler"]["max_lr"] = suggested_lr
        print(f"Suggested maximum learning rate: {suggested_lr}")
        return # only learning rate finder is enabled, no training is done
    else:
        print("-----Learning rate finder is disabled.")

    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['scheduler']['max_lr']/config['training']['scheduler']['div_factor'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader) * config['training']['epochs']

    # Define scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['scheduler']['max_lr'],
        total_steps=total_steps,
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy='cos',
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor']
    )

    criterion = torch.nn.CrossEntropyLoss()

    mix_precision = config["training"].get("mix_precision", False)
    if mix_precision:
        if device.type == "cuda":
            scaler = GradScaler('cuda')
        elif device.type == "mps":
            scaler = GradScaler('mps') 
        else:
            scaler = None
            mix_precision = False
    else:
        scaler = None
        mix_precision = False

    if mix_precision:
        print("Mixed precision training enabled.")

    log_dir = os.path.join(config['training']['log_dir'], f"{config['data']['dataset']}_{config['model']['name']}")
    metric_logger = MetricLogger(log_dir)
    print(f"-----Logging metrics to {log_dir}")

    resume_training = True
    start_epoch = 0
    checkpoint_dir = os.path.join("checkpoints", f"{config['data']['dataset']}_{config['model']['name']}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    print(f"-----Checkpoints will be saved to {checkpoint_dir}, and will be loaded from {checkpoint_path}")

    if resume_training and os.path.exists(checkpoint_path):
        print(f"-----Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    else:
        print(f"-----No checkpoint found at {checkpoint_path}, starting from scratch.")


    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        train_state_dict = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, scaler)
        train_state_dict['epoch'] = epoch+1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"-----Checkpoint saved to {checkpoint_dir}")
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_{epoch}.pth"))
        print(f"-----Model saved to {checkpoint_dir}")
        
        metric_logger.log_metrics(train_state_dict)
        
        test_state_dict = test(model, test_loader, criterion, device, mix_precision)
        test_state_dict['epoch'] = epoch+1
        metric_logger.log_metrics(test_state_dict)

        print(f"Train Loss: {train_state_dict['loss']:.4f}, Train Acc: {train_state_dict['accuracy']:.2f}%")
        print(f"Test Loss: {test_state_dict['loss']:.4f}, Test Acc: {test_state_dict['accuracy']:.2f}%")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{config['data']['dataset']}_{config['model']['name']}.pth"))
    print(f"-----Model saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()


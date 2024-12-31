# ResNet50 Training on ImageNet

This repository contains code for training ResNet50 on ImageNet using PyTorch. The implementation supports multiple device types (CUDA, MPS, CPU) and includes features like mixed-precision training and learning rate scheduling.

## Live Demo

Try out the trained model on Hugging Face Spaces:
[ResNet50 ImageNet Demo](https://huggingface.co/spaces/peeyushsinghal/resnet50_imagenet)

Features:
- Upload your own images
- Get real-time predictions
- View top-5 class predictions with confidence scores
- Interactive web interface

## Dataset

We use the ImageNet-1K dataset which can be accessed through:
1. Official ImageNet website (requires registration): [image-net.org](https://image-net.org/)
2. Academic Torrents: [ImageNet torrent](https://academictorrents.com/details/943977d8c96892d24237638335e481f3ccd54cfb)

- **Full Dataset Specs:**
  - 1000 classes
  - Training: ~1.2M images
  - Validation: 50,000 images
  - Image size: Variable (commonly resized to 224x224)
  - Format: RGB

## Training Strategies

### Learning Rate Policy
We implement the One Cycle Policy as described in Leslie Smith's paper ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"](https://arxiv.org/abs/1708.07120).

#### One Cycle Policy Details:
- Learning rate increases from `initial_lr` to `max_lr` for 30% of training
- Then decreases from `max_lr` to `min_lr` for 70% of training
- Momentum decreases from 0.95 to 0.85 during the first 30%
- Momentum increases from 0.85 to 0.95 during the remaining 70%

```yaml
scheduler:
  name: onecycle
  max_lr: 0.1
  epochs: 30
  steps_per_epoch: 1024  # depends on batch_size and dataset size
  pct_start: 0.3        # percentage of iterations for lr warmup
  anneal_strategy: 'cos' # cosine annealing
```

### 1. 10% ImageNet Subset (Quick Experimentation)
For rapid prototyping and initial experiments, using 10% of ImageNet is recommended:
- Training: ~120,000 images
- Validation: 5,000 images
- Benefits:
  - Faster iteration cycles
  - Lower storage requirements (~13GB vs 130GB)
  - Quicker model validation
- Recommended settings in `config_imagenet.yml`:
  ```yaml
  training:
    epochs: 30  # Reduced epochs for faster iteration
    batch_size: 256 # depends on the GPU
    subset_fraction: 0.1  # Uses 10% of data
  
  scheduler:
    name: onecycle
    max_lr: 0.1
    pct_start: 0.3
    anneal_strategy: 'cos'
  ```
- Expected training time: ~4-6 hours on g6.xlarge

### 2. Full ImageNet Training
For final model training and state-of-the-art results:
- Training: All 1.2M images
- Validation: All 50,000 images
- Recommended settings in `config_imagenet.yml`:
  ```yaml
  training:
    epochs: 30 
    batch_size: 256
    subset_fraction: 1.0  # Uses full dataset
  
  scheduler:
    name: onecycle
    max_lr: 0.1
    pct_start: 0.3
    anneal_strategy: 'cos'
  ```
- Expected training time: ~100 hours on g6.xlarge

## Performance Comparison

| Dataset Version | Top-1 Accuracy | Top-5 Accuracy | Training Time | Learning Rate Strategy |
|----------------|---------------|---------------|---------------|---------------------|
| 10% Subset     | ~60-65%       | ~82-85%       | 4-6 hours    | One Cycle Policy    |
| Full Dataset   | ~76-78%       | ~93-94%       | 80-120 hours| One Cycle Policy    |

## Usage

1. Prepare ImageNet data:
```bash
# For 10% subset training
python main.py --config config_imagenet.yml --subset 0.1
# For full dataset training
python main.py --config config_imagenet.yml
```

## Best Practices for Subset Training

1. **Class Balance**: The subset maintains class distribution of the original dataset
2. **Random Seed**: Fix random seed for reproducible subset selection
3. **Validation**: Use a proportional subset of validation data
4. **Hyperparameters**:
   - Use One Cycle Policy for faster convergence
   - Find optimal max_lr using lr_finder
   - Adjust batch size based on available GPU memory
   - Consider smaller model variations for faster iteration

## Learning Rate Finder
To find the optimal maximum learning rate:
```bash
python lr_finder.py --config config_imagenet.yml
```
This will:
1. Plot learning rate vs loss
2. Suggest optimal learning rate range
3. Help configure the One Cycle Policy

## Model Checkpointing

- Checkpoints are saved every epoch for subset training
- For full dataset, saves every 10 epochs
- Best model saved based on validation accuracy
- Resume training supported via checkpoint loading

## Model Deployment

Visit the demo: [ResNet50 ImageNet Demo](https://huggingface.co/spaces/peeyushsinghal/resnet50_imagenet)

## References

```bibtex
@article{smith2018superconvergence,
  title={Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates},
  author={Smith, Leslie N and Topin, Nicholay},
  journal={arXiv preprint arXiv:1708.07120},
  year={2018}
}
```


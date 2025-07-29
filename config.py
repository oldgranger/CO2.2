from typing import Dict, Any

import torch


class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data configuration
    DATA_PATH = 'datasets/data'
    IMG_SIZE = 640
    BATCH_SIZE = 2
    NUM_WORKERS = 0

    # Model configuration
    NUM_CLASSES = 3
    CLASS_NAMES = ["Paper", "Rock", "Scissors"]
    HIDDEN_DIM = 256
    NHEADS = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6

    # Training configuration
    NUM_EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PRINT_FREQ = 10

    # Loss weights
    LOSS_WEIGHTS: Dict[str, float] = {
        'loss_ce': 1,
        'loss_bbox': 5,
        'loss_giou': 2
    }

    # Matcher costs
    MATCHER_COSTS: Dict[str, float] = {
        'cost_class': 1,
        'cost_bbox': 1,
        'cost_giou': 1
    }

    # EOS coefficient
    EOS_COEF = 0.1


config = Config()
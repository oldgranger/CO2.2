from typing import Dict, Any

import torch


class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    WARMUP_EPOCHS = 3
    MAX_GRAD_NORM = 0.1
    USE_TENSORBOARD = True

    DATA_PATH = 'datasets/data'
    IMG_SIZE = 640
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    # Model configuration
    NUM_CLASSES = 3
    CLASS_NAMES = ["Paper", "Rock", "Scissors"]
    HIDDEN_DIM = 192
    NHEADS = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    # Training configuration
    NUM_EPOCHS = 20
    LR = 3e-5
    WEIGHT_DECAY = 1e-3
    PRINT_FREQ = 10

    # Loss weights
    LOSS_WEIGHTS: Dict[str, float] = {
        'loss_ce': 1,
        'loss_bbox': 5,
        'loss_giou': 5
    }

    # Matcher costs
    MATCHER_COSTS: Dict[str, float] = {
        'cost_class': 1,
        'cost_bbox': 5,
        'cost_giou': 2
    }

    # EOS coefficient
    EOS_COEF = 0.1

    DROPOUT = 0.3
    LR_SCHEDULER = {
        'factor': 0.5,
        'patience': 1,
        'min_lr': 1e-5
    }

    USE_AMP = True
    PATIENCE = 5
    CHECKPOINT_DIR = 'checkpoints/'

config = Config()
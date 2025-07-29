import os
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from config import config
from data.dataset import load_datasets
from data.collate import collate_fn
from models.rfdetr import RFDETR
from losses.matcher import HungarianMatcher
from losses.criterion import SetCriterion
from train import train_model
from torch.optim import AdamW

def main():
    # Load data
    train_dataset, val_dataset, test_dataset = load_datasets(config.DATA_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # Initialize model
    model = RFDETR(
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        nheads=config.NHEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS
    )

    # Initialize matcher and criterion
    matcher = HungarianMatcher(**config.MATCHER_COSTS)
    criterion = SetCriterion(
        config.NUM_CLASSES,
        matcher,
        config.LOSS_WEIGHTS,
        eos_coef=config.EOS_COEF
    ).to(config.DEVICE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Debug: Test forward pass
    print("Testing forward pass with dummy data...")
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    dummy_output = model(dummy_input)
    print("Dummy forward pass successful!")

    # Train model
    print("Starting training...")
    model = train_model(
        model, criterion, optimizer,
        train_loader, val_loader,
        num_epochs=config.NUM_EPOCHS
    )

    print("Training completed!")
    torch.save(model.state_dict(), "rfdetr_final.pth")
    print("✅ Model weights saved to rfdetr_final.pth")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
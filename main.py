import torch
from torch.utils.data import DataLoader
from config import config
from data.dataset import load_datasets
from data.collate import collate_fn
from models.rfdetr import RFDETR
from losses.matcher import HungarianMatcher
from losses.criterion import SetCriterion
from train import train_model
from torch.optim import AdamW
from evaluate import evaluate_coco
from utils.visualize import plot_predictions


def main():
    # Setup device and reproducibility
    torch.backends.cudnn.benchmark = False
    print(f"\nUsing device: {config.DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Load data
    train_dataset, val_dataset, test_dataset = load_datasets(config.DATA_PATH)

    # Verify data
    sample_img, sample_target = train_dataset[0]
    print("=== Data Sample ===")
    print(f"Image shape: {sample_img.shape}")
    print(f"Boxes: {sample_target['boxes'].shape}")
    print(f"Labels: {sample_target['labels']}\n")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(2, config.NUM_WORKERS),  # Safer for Windows
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(2, config.NUM_WORKERS),
        pin_memory=True
    )

    # Initialize model
    model = RFDETR(
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        nheads=config.NHEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS
    ).to(config.DEVICE)

    # Loss and optimizer
    matcher = HungarianMatcher(**config.MATCHER_COSTS)
    criterion = SetCriterion(
        config.NUM_CLASSES,
        matcher,
        config.LOSS_WEIGHTS,
        eos_coef=config.EOS_COEF
    ).to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Train
    print("=== Starting Training ===")
    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS
    )

    # Save and evaluate
    torch.save(model.state_dict(), "rfdetr_final.pth")
    print("\n=== Evaluation ===")
    plot_predictions(model, val_dataset, config.DEVICE, n=3)

    if hasattr(val_dataset, 'coco_json_path'):
        coco_stats = evaluate_coco(model, val_loader, config.DEVICE, val_dataset.coco_json_path)
        print(f"\nmAP@0.5: {coco_stats[1]:.4f} | mAP@0.5:0.95: {coco_stats[0]:.4f}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    main()
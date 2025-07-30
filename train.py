from tqdm import tqdm
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
from evaluate import evaluate


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scaler=None, print_freq=10):
    model.train()
    criterion.train()

    running_loss = 0.0
    loss_components = {
        'ce': 0.0,
        'bbox': 0.0,
        'giou': 0.0
    }

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")

    for batch_idx, (samples, targets) in progress_bar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

            # Calculate weighted loss
            losses = sum(loss_dict[k] * config.LOSS_WEIGHTS[k] for k in loss_dict.keys())

        # Skip problematic batches
        if not torch.isfinite(losses):
            print(f"Invalid loss in batch {batch_idx}, skipping...")
            continue

        # Backward pass
        optimizer.zero_grad()
        if scaler:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

        # Update metrics
        running_loss += losses.item()
        for k in loss_components:
            loss_components[k] += loss_dict[f'loss_{k}'].item()

        # Update progress bar
        if batch_idx % print_freq == 0:
            progress_bar.set_postfix({
                'Loss': losses.item(),
                'CE': loss_dict['loss_ce'].item(),
                'BBox': loss_dict['loss_bbox'].item(),
                'GIoU': loss_dict['loss_giou'].item()
            })

    # Calculate epoch averages
    avg_loss = running_loss / len(data_loader)
    print(f"\nEpoch {epoch} Training Summary:")
    print(f"Total Loss: {avg_loss:.4f}")
    for k, v in loss_components.items():
        print(f"{k.upper()} Loss: {v / len(data_loader):.4f}")

    return avg_loss


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    model.to(config.DEVICE)
    best_metrics = {
        'val_loss': float('inf'),
        'epoch': -1
    }

    # Setup checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=getattr(config, 'LR_SCHEDULER_FACTOR', 0.1),
        patience=getattr(config, 'LR_SCHEDULER_PATIENCE', 2),
        verbose=True
    )

    # Initialize AMP scaler if available
    use_amp = getattr(config, 'USE_AMP', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        # Train phase
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer,
            config.DEVICE, epoch, scaler
        )

        # Validation phase
        val_loss = evaluate(model, criterion, val_loader, config.DEVICE)
        print(f"\nValidation Loss: {val_loss:.4f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save checkpoints
        is_best = val_loss < best_metrics['val_loss']
        if is_best:
            best_metrics.update({
                'val_loss': val_loss,
                'epoch': epoch
            })

        # Always save latest and best models
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_metrics['val_loss']
        }, f'checkpoints/latest.pth')

        if is_best:
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"New best model saved at epoch {epoch} with val loss: {val_loss:.4f}")

        # Early stopping check
        if epoch - best_metrics['epoch'] > config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining completed. Best val loss: {best_metrics['val_loss']:.4f} at epoch {best_metrics['epoch']}")
    return model
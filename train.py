from tqdm import tqdm
import torch
from config import config
from evaluate import evaluate


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, print_freq=10):
    model.train()
    criterion.train()

    running_loss = 0.0
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}")

    for batch_idx, (samples, targets) in progress_bar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        if any(torch.isnan(v) or torch.isinf(v) for v in loss_dict.values()):
            print(f"⚠️  NaN in batch {batch_idx}:")
            for k, v in loss_dict.items():
                print(f"  {k}: {v}")
            continue

        losses = sum(loss_dict.values())
        loss_value = losses.item()

        if not torch.isfinite(losses):
            print(f"❌ Loss is not finite at batch {batch_idx}. Skipping.")
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += loss_value
        if batch_idx % print_freq == 0:
            progress_bar.set_postfix(loss=loss_value)

    avg_loss = running_loss / len(data_loader)
    print(f"✅ Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    model.to(config.DEVICE)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, config.DEVICE, epoch)
        val_loss = evaluate(model, criterion, val_loader, config.DEVICE)

        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model
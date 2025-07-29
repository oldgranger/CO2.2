import torch
from tqdm import tqdm
from config import config

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            total_loss += losses.item()

    return total_loss / len(data_loader)
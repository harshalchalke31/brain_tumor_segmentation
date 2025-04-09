import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.utils.data import Dataset
import segmentation_models_3D as sm
from tqdm import tqdm

class BraTsDataset(Dataset):
    def __init__(self, img_dir:str, img_list:list, mask_dir:str, mask_list:list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = torch.load(self.img_dir + self.img_list[idx])
        mask = torch.load(self.mask_dir + self.mask_list[idx])
        return img, mask

def train_UNet_multiclass(model, train_loader, valid_loader, device: str, num_classes: int,
                          num_epochs: int = 500, lr: float = 1e-3,
                          log_path: str = './logs/train_log_multiclass.csv',
                          model_path: str = './models/model1/best_model_multiclass.pth',
                          patience: int = 50,
                          losses: str = None):

    if losses is not None:
        losses = losses.upper()

    if losses == 'DF':
        diceloss = sm.losses.DiceLoss()
        focalloss = sm.losses.CategoricalFocalLoss()
        criterion = diceloss + focalloss
    elif losses == 'DCE':
        diceloss = sm.losses.DiceLoss()
        celoss = sm.losses.CategoricalCELoss()
        criterion = diceloss + celoss
    else:
        criterion = CrossEntropyDiceLoss(num_classes=num_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_loss = float('inf')
    best_dice = 0.0
    patience_counter = 0
    model.to(device)

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Valid Loss", "Valid Dice"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_size_curr = images.size(0)
            train_loss += loss.item() * batch_size_curr
            total_train_samples += batch_size_curr

        train_loss /= total_train_samples

        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        total_valid_samples = 0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False):
                images = images.to(device)
                masks = masks.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, masks)

                batch_size_curr = images.size(0)
                valid_loss += loss.item() * batch_size_curr
                total_valid_samples += batch_size_curr

                batch_dice = multiclass_dice_coefficient(outputs, masks, num_classes)
                valid_dice += batch_dice * batch_size_curr

        valid_loss /= total_valid_samples
        valid_dice /= total_valid_samples
        scheduler.step()

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, valid_loss, valid_dice])

        # Save model on best loss or best dice
        if valid_loss < best_loss or valid_dice > best_dice:
            torch.save(model.state_dict(), model_path)
            best_loss = min(valid_loss, best_loss)
            best_dice = max(valid_dice, best_dice)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid Dice: {valid_dice:.4f} | "
              f"Patience: {patience_counter}")

        if patience_counter >= patience:
            print("Early Stopping Triggered!")
            break



class DiceLossMulticlass(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLossMulticlass, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets, num_classes):
        preds = torch.softmax(preds, dim=1)  # [B, C, D, H, W]
        dice = 0.0
        for c in range(num_classes):
            pred_c = preds[:, c]
            target_c = (targets == c).float()  # One-hot per class
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice += (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice / num_classes

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.dice = DiceLossMulticlass()

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)  # targets are class indices [B, D, H, W]
        dice_loss = self.dice(preds, targets, self.num_classes)
        return ce_loss + dice_loss

def multiclass_dice_coefficient(preds, targets, num_classes, smooth=1e-5):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)

    dice = 0.0
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return (dice / num_classes).item()

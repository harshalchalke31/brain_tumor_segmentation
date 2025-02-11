import torch.nn as nn
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2
import csv
import random
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    
    def forward(self,pred,target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred*target).sum()
        return 1- ((2.* intersection +smooth)/(pred.sum()+target.sum()+smooth))

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)  # Works with raw logits
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Compute BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Compute Dice loss; apply sigmoid to get probabilities first
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return bce_loss + dice_loss

def dice_coefficient(preds, targets, smooth=1e-5, threshold=0.5):
    # Apply sigmoid to logits and then threshold to get binary masks
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    # Flatten the tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None,preload=False):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.transform = transform
        self.preload = preload

        if self.preload:
            print("Preloading dataset into memory...")
            self.images = [self.load_image(path) for path in self.image_paths]
            self.masks = [self.load_mask(path) for path in self.mask_paths]


    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def load_mask(self, path):
        mask = Image.open(path).convert("L")
        return mask
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.preload:
            image = self.images[idx]
            mask = self.masks[idx]
        else:
            image = self.load_image(self.image_paths[idx])
            mask = self.load_mask(self.mask_paths[idx])


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

class CustomDataLoader:
    """
    A simple, minimal custom dataloader that:
      - Can shuffle dataset each epoch
      - Yields batches of data from the dataset
      - Implements the iterator protocol: __iter__ and __next__

    Example usage:
        my_loader = CustomDataLoader(dataset, batch_size=16, shuffle=True)
        for epoch in range(10):
            for batch_data, batch_labels in my_loader:
                # Training code here...
    """

    def __init__(self, dataset, batch_size=16, shuffle=True):
        """
        Args:
            dataset: An object implementing __len__ and __getitem__(idx)
            batch_size (int): How many samples per batch
            shuffle (bool): Whether to shuffle at the start of each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Precompute dataset size and an index tensor
        self.dataset_size = len(self.dataset)
        self.indices = torch.arange(self.dataset_size)

        # Compute number of total batches (ceiling division)
        self.num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

        # Track the current batch index during iteration
        self.current_batch = 0

    def __iter__(self):
        """
        Called at the start of an iteration (e.g. `for batch in loader`).
        Resets the current batch index to 0 and shuffles indices if needed.
        Returns self (the iterator).
        """
        if self.shuffle:
            # Shuffle indices in-place
            shuffled = torch.randperm(self.dataset_size)
            self.indices = self.indices[shuffled]

        # Reset the batch pointer
        self.current_batch = 0
        return self

    def __next__(self):
        """
        Called each time we request the next batch (e.g. `next(loader_iter)`).
        Raises StopIteration if we've exhausted all batches.
        """
        if self.current_batch >= self.num_batches:
            # No more batches to yield
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        batch_data = []
        batch_labels = []
        for idx in batch_indices:
            x, y = self.dataset[idx]
            batch_data.append(x)
            batch_labels.append(y)

        # Stack into tensors
        batch_data = torch.stack(batch_data)
        batch_labels = torch.stack(batch_labels)

        self.current_batch += 1
        return batch_data, batch_labels

    def __len__(self):
        """Return the total number of batches (not samples)."""
        return self.num_batches
    
def plot_loss_curves(log_path:Path):
    df = pd.read_csv(log_path)  

    # Extract columns
    epochs = df['Epoch']
    train_loss = df['Train Loss']
    valid_loss = df['Valid Loss']
    valid_dice = df['Valid Dice']

    # Create a figure with two subplots side by side
    plt.figure(figsize=(10, 4))

    # --- Subplot 1: Loss curves ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # --- Subplot 2: Validation Dice ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_dice, label='Valid Dice', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Validation Dice')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

class UNetTester:
    def __init__(self, 
                 model, 
                 device,
                 model_path, 
                 test_dataset,
                 output_dir="./test_results",
                 csv_filename="test_metrics.csv"):
        self.model = model
        self.device = device
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, csv_filename)

        # We'll make a custom loader with batch_size=1 for convenience:
        self.test_loader = CustomDataLoader(test_dataset, batch_size=1, shuffle=False)

    def load_weights(self):
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def compute_sample_metrics(self, logits, targets, threshold=0.5, smooth=1e-5):
        """
        Compute segmentation metrics between predicted (raw logits) and targets.
        Returns dict with dice, jaccard, precision, recall, f1
        """
        # 1) Convert logits -> binary predictions
        probs = torch.sigmoid(logits)  # shape [1, 1, H, W]
        preds = (probs > threshold).long()
        targets = targets.long()

        # 2) Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum().float()
        union = (preds_flat + targets_flat).clamp_max(1).sum().float()  # or bitwise OR

        # 3) Compute
        dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
        jaccard = (intersection + smooth) / (union + smooth)

        total_pred_positive = preds_flat.sum().float()
        precision = (intersection + smooth) / (total_pred_positive + smooth)

        total_gt_positive = targets_flat.sum().float()
        recall = (intersection + smooth) / (total_gt_positive + smooth)

        f1 = (2 * precision * recall) / (precision + recall + smooth)

        return {
            "dice": dice.item(),
            "jaccard": jaccard.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item()
        }

    def evaluate_model(self):
        """
        1) Evaluate the model on the entire test set.
        2) Compute sample-wise metrics, save them to CSV.
        3) Return all metrics for further usage or summarizing.
        """
        self.load_weights()  # ensure weights are loaded
        # Prepare CSV
        with open(self.csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SampleID", "Dice", "Jaccard", "Precision", "Recall", "F1"])

        all_metrics = []

        for idx, (image, mask) in enumerate(self.test_loader):
            image = image.to(self.device)   # shape [1, 3, H, W]
            mask  = mask.to(self.device)    # shape [1, 1, H, W]

            with torch.no_grad():
                logits = self.model(image)  # raw model outputs

            metrics_dict = self.compute_sample_metrics(logits, mask)
            all_metrics.append(metrics_dict)

            # Save each sample's metrics to CSV
            with open(self.csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx+1,
                    metrics_dict["dice"],
                    metrics_dict["jaccard"],
                    metrics_dict["precision"],
                    metrics_dict["recall"],
                    metrics_dict["f1"],
                ])

        # Compute average metrics
        avg_metrics = {
            "dice":     sum(m["dice"]     for m in all_metrics)/len(all_metrics),
            "jaccard":  sum(m["jaccard"]  for m in all_metrics)/len(all_metrics),
            "precision":sum(m["precision"]for m in all_metrics)/len(all_metrics),
            "recall":   sum(m["recall"]   for m in all_metrics)/len(all_metrics),
            "f1":       sum(m["f1"]       for m in all_metrics)/len(all_metrics),
        }

        # Append average row in CSV
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Average",
                avg_metrics["dice"],
                avg_metrics["jaccard"],
                avg_metrics["precision"],
                avg_metrics["recall"],
                avg_metrics["f1"],
            ])

        print(f"Per-sample and average metrics saved to: {self.csv_path}")
        return all_metrics, avg_metrics

    def visualize_predictions(self, num_samples=10):
        """
        Visualize predictions for a random subset of test samples.
        Saves images [MRI, True Mask, Predicted Mask] as .png files.
        """
        # Pick random samples
        random_indices = random.sample(range(len(self.test_dataset)), min(num_samples, len(self.test_dataset)))

        for sample_idx in random_indices:
            image, mask = self.test_dataset[sample_idx]  # image [3,H,W], mask [1,H,W]
            image_tensor = image.unsqueeze(0).to(self.device)  # [1,3,H,W]
            
            with torch.no_grad():
                logits = self.model(image_tensor)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float().cpu().squeeze().numpy()

            # Convert to numpy for display
            image_np = image.permute(1,2,0).cpu().numpy()  # shape [H, W, 3]
            mask_np  = mask.squeeze().cpu().numpy()        # shape [H, W]

            # Plot them
            fig, axs = plt.subplots(1, 3, figsize=(12,4))
            axs[0].imshow(image_np, cmap='gray')
            axs[0].set_title("MRI Image")
            axs[0].axis('off')

            axs[1].imshow(mask_np, cmap='gray')
            axs[1].set_title("True Mask")
            axs[1].axis('off')

            axs[2].imshow(pred_mask, cmap='gray')
            axs[2].set_title("Predicted Mask")
            axs[2].axis('off')

            fig.tight_layout()
            out_path = os.path.join(self.output_dir, f"sample_{sample_idx}.png")
            plt.savefig(out_path)
            plt.close(fig)

        print(f"Saved {len(random_indices)} prediction visualizations to {self.output_dir}")


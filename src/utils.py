import torch.nn as nn
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2

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

def custom_dataloader(dataset, batch_size=16, shuffle=True):
    """
    A generator function that yields batches of data from the dataset
    without using torch.utils.data.DataLoader.
    """
    dataset_size = len(dataset)
    indices = torch.arange(dataset_size)

    if shuffle:
        indices = indices[torch.randperm(dataset_size)]

    # Go through indices in mini-batch increments
    for start_idx in range(0, dataset_size, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        
        # Collect the data samples for the current batch
        batch_data = []
        batch_labels = []
        for i in batch_indices:
            x, y = dataset[i]
            batch_data.append(x)
            batch_labels.append(y)
        
        # Convert list of tensors to a single tensor (optional)
        batch_data = torch.stack(batch_data)     # shape: (batch_size, channels, height, width)
        batch_labels = torch.stack(batch_labels) # shape: (batch_size,)
        
        yield batch_data, batch_labels


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
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np


class SignatureDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None, target_size=(640, 640)):
        """
        Simple dataset for signature detection
        
        Args:
            csv_file: Path to CSV file with format: filename,label,x1,y1,x2,y2
            images_dir: Directory containing the images
            transform: Optional transforms
            target_size: Target image size (width, height)
        """
        self.annotations = pd.read_csv(csv_file, names=['filename', 'label', 'x1', 'y1', 'x2', 'y2'])
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get annotation
        row = self.annotations.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, row['filename'])
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # use PIL resize... not resize as transform ... it is necessary step for each model even baseline
        image = image.resize(self.target_size)
        
        # Get bounding box and normalize to [0, 1]
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        # Normalize coordinates to [0,1] range
        norm_x1 = x1 / original_size[0]  
        norm_y1 = y1 / original_size[1]
        norm_x2 = x2 / original_size[0]
        norm_y2 = y2 / original_size[1]
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        else: # baseline model without any transforms
            image = transforms.ToTensor()(image)
            
        bbox = torch.tensor([norm_x1, norm_y1, norm_x2, norm_y2], dtype=torch.float32)
        
        return image, bbox


class SignatureDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, test_csv, images_dir, batch_size=4, num_workers=0, target_size=(640, 640)):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        
        # transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def setup(self, stage=None):
        """Setup datasets"""
        if stage == 'fit' or stage is None:
            self.train_dataset = SignatureDataset(
                self.train_csv, 
                self.images_dir, 
                transform=self.transform,
                target_size=self.target_size
            )
            
            self.val_dataset = SignatureDataset(
                self.test_csv, 
                self.images_dir, 
                transform=self.transform,
                target_size=self.target_size
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import box_iou, distance_box_iou_loss, complete_box_iou_loss


class SimpleBaseModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),  # 640 -> 320
            nn.ReLU(),
            nn.MaxPool2d(2),  # 320 -> 160
            
            # Second conv block  
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 160 -> 80
            nn.ReLU(),
            nn.MaxPool2d(2),  # 80 -> 40
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20 -> 10
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 10 -> 5
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5 -> 2 (approximately)
        )
        
        # Regression head for bounding box prediction
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),  # Approximate size after convolutions
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)  # x1, y1, x2, y2
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Predict bounding box
        bbox = self.bbox_head(features)
        
        # Apply sigmoid to keep coordinates in [0, 1] range
        bbox = torch.sigmoid(bbox)
        
        return bbox
    
    def training_step(self, batch, batch_idx):
        images, target_bboxes = batch
        
        # Forward pass
        predicted_bboxes = self(images)
        
        # Compute Complete IoU loss (better than Distance IoU)
        loss = complete_box_iou_loss(predicted_bboxes, target_bboxes, reduction='mean')
        
        # metrics
        iou = box_iou(predicted_bboxes, target_bboxes).diag().mean()
        
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, target_bboxes = batch
        
        # Forward pass
        predicted_bboxes = self(images)
        
        # Compute Complete IoU loss
        loss = complete_box_iou_loss(predicted_bboxes, target_bboxes, reduction='mean')
        
        # Metrics
        iou = box_iou(predicted_bboxes, target_bboxes).diag().mean()
        
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

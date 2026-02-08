import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import box_iou, distance_box_iou_loss, complete_box_iou_loss
from torchvision.models import mobilenet_v3_large


class MobileNetModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.freeze_backbone = freeze_backbone
        
        # MobileNetV3 Large backbone
        self.backbone = mobilenet_v3_large(pretrained=pretrained)
        
        # Get feature extractor (remove classifier)
        self.features = self.backbone.features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # MobileNetV3 Large outputs 960 features from the last conv layer
        backbone_output_size = 960
        
        # Regression head for bounding box prediction
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(backbone_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # x1, y1, x2, y2
        )
        
    def forward(self, x):
        # Extract features using MobileNetV3 feature extractor
        features = self.features(x)
        
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
        
        # Debug the loss and IoU values
        if batch_idx == 0:
            print(f"  Computed loss: {loss.item():.6f}")
            print(f"  Computed IoU: {iou.item():.6f}")
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use different learning rates for backbone and head if backbone is not frozen
        if not self.freeze_backbone:
            # Lower learning rate for pretrained backbone
            backbone_params = list(self.features.parameters())
            head_params = list(self.bbox_head.parameters())
            
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # 10x lower for backbone
                {'params': head_params, 'lr': self.learning_rate}
            ])
        else:
            # Only optimize the head if backbone is frozen
            optimizer = torch.optim.Adam(self.bbox_head.parameters(), lr=self.learning_rate)
        
        return optimizer
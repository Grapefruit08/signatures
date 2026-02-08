"""
Fine tuning YOLOv8n model
"""
from ultralytics import YOLO
import torch
import os
import albumentations as A


def train_yolo_signatures(dataset_path: str, use_custom_transforms: bool = False) -> tuple[str, dict]:
    """Train YOLOv8n on signature dataset"""
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')  # Will download if not exists
    
    # Training with custom Albumentations transforms
    custom_transforms = None
    if use_custom_transforms:
        custom_transforms = A.Compose([
            # Geometric transformations
            A.Rotate(limit=90, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.5, rotate_limit=0, p=0.6),  # Translation + scaling
            A.HorizontalFlip(p=0.5),
        
            # Intensity transformations  
            A.Blur(blur_limit=7, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=20, p=0.4),
        ])
    
    # Training configuration - minimal, focused on fine-tuning
    training_config = {
        'data': os.path.join(dataset_path, 'dataset.yaml'),
        'epochs': 10,           # A few more epochs since we're adding augmentations
        'batch': 16,
        'imgsz': 640,
        'device': device,
        'project': 'signature_detection',
        'name': 'yolov8n_signatures_non_augmented',
        'freeze': 10,           # Freeze backbone layers (only train detection head)
        'lr0': 0.01,            # Higher LR since only training head
        'patience': 5,          # Early stopping
        'val': True,
        'plots': True,
        'save_json': True,
        
        # Data augmentation
        #'degrees': 90.0,        # Random rotation degrees
        #'translate': 0.1,       # Random translation of image size
        #'scale': 0.5,           # Random scaling from 50% to 150%
        #'flipud': 0.0,          # Vertical flip probability
        #'fliplr': 0.5,          # Horizontal flip           
    }


    # Train the model
    try:
        if use_custom_transforms:
            train_results = model.train(
                **training_config,
                augmentations=custom_transforms,
            )
        else:
            train_results = model.train(
                **training_config
            )
        print("\nTraining completed successfully!\n")
        print("\n Training Results: ")
        print(f"Train mAP50: {train_results.box.map50:.4f}")
        print(f"Train mAP50-95: {train_results.box.map:.4f}")

        
        # Get the best model path
        best_model_path = model.trainer.best
        print(f"Best model saved at: {best_model_path}")
        
        # Validate the model
        print("Running validation...")
        val_results = model.val()
        print("\n Validation Results: ")
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Validation mAP50-95: {val_results.box.map:.4f}")
        
        # Export the model (optional)
        print("Exporting model...")
        model.export(format='onnx')  # Export to ONNX format
        print("Model exported to ONNX format")
        
        return best_model_path, val_results
        
    except Exception as e:
        print("Training failed with error:")
        print("Exception: ", e)
        return None, None


if __name__ == "__main__":
    # Check if dataset exists
    dataset_path = "yolo_signature_dataset"
    
    # Train the model
    best_model, val_results = train_yolo_signatures(
        dataset_path=dataset_path,
        use_custom_transforms=False,  # Use albumentations transforms
    )
    
    # Test the model if training was successful
    if best_model:
        print(f"Best model path: {best_model}")
        print(f"Validation results: {val_results}")

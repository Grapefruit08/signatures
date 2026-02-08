# Signature Detection Project

This project implements a signature detection system using YOLO (You Only Look Once) object detection model.

## CLI Usage

The command-line interface is available through `cli_detect.py` for easy signature detection on documents.

```bash
python cli_detect.py [arguments]
```

## Model Architecture

### YOLO Model Selection

I chose the YOLO model architecture with a transfer learning approach:
- **Frozen layers**: Most of the pre-trained YOLO layers are frozen
- **Fine-tuned layers**: Only the last layers are trained on the signature detection task
- **Training code**: Located in `train_yolo_signatures.py`
- **Dataset conversion**: Transformed Kaggle dataset to YOLO standard format using `convert_to_yolo.py`

### Training Configuration

The model was trained with minimal epochs for efficiency:
- **Epochs**: 10 (small number for quick training)
- **Best results**: Achieved with basic settings without additional augmentations

## Data Augmentation Experiments

### Transforms Tested

1. **Geometric Transforms**: Tried built-in geometrical transformations
2. **Intensity Transforms**: Experimented with intensity-based augmentations
3. **Final Decision**: Best results obtained with **basic settings without any added transforms and augmentations**

### Model Performance Results

Tested all mentioned YOLO models on the dataset:
- **Best performing model**: YOLO without augmentation
- **Performance metrics**:
  - 2 false detections in background samples
  - 5 true detections in signature samples

## Alternative Approaches

### Custom Architecture with PyTorch Lightning

Initially attempted to build a custom architecture using the PyTorch Lightning library:
- **Challenge**: Unable to train the model sufficiently with this approach
- **Decision**: Switched to YOLO model instead

### YOLO Success

The YOLO model provided good results even without:
- Additional augmentations
- Translation transforms
- Extensive fine-tuning

This demonstrates the effectiveness of the pre-trained YOLO architecture for signature detection tasks.

## Project Structure

- `cli_detect.py` - Command-line interface for signature detection
- `train_yolo_signatures.py` - YOLO model training implementation
- `training_config.yaml` - Training configuration settings
- Various dataset folders with signatures and background samples
- Experiment folders documenting different training approaches

## TODO

- **Configuration Management**: Replace hardcoded values in YOLO training with configuration loaded from YAML file for better maintainability and flexibility
- **Inference Optimization**: Use ONNX Runtime for inference to improve performance and reduce dependencies
- **Extended Training**: 
* Train YOLO model for more epochs - with longer training
* Augmentation techniques then could potentially help improve model performance without degrading results
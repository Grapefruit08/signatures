"""
Convert Kaggle signature dataset to YOLO format
"""
import os
import pandas as pd
from PIL import Image


def create_yolo_dataset_yaml(output_dir: str):
    """
    Create dataset YAML file for YOLO
    """
    dataset_yaml = f"""
# Signature Detection Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['signature']  # class names
"""
    
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        f.write(dataset_yaml)


def convert_bbox_to_yolo(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_width: int, 
        img_height: int
    ) -> tuple[float, float, float, float]:
    """
    Convert absolute bounding box to YOLO format (normalized center + width/height)
    """
    # Calculate center coordinates
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Calculate width and height
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Normalize to [0, 1]
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm


def convert_to_yolo(
        input_dir: str,
        bbox_dataframe: pd.DataFrame, 
        output_dir: str, 
        output_folder_name: str
    ):
    """
    Convert bounding boxes to YOLO format and save images and labels
    """
    for _, row in bbox_dataframe.iterrows():
        filename = row['filename']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        # Load image
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Convert to YOLO format
        center_x, center_y, width, height = convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height)
        
        # Convert .tif to .jpg for YOLO compatibility
        img_name = filename.replace('.tif', '.jpg')
        img.convert('RGB').save(f"{output_dir}/images/{output_folder_name}/{img_name}", 'JPEG')
        
        # Create label file
        label_name = filename.replace('.tif', '.txt')
        with open(f"{output_dir}/labels/{output_folder_name}/{label_name}", 'w') as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def create_yolo_dataset(input_dir: str, output_dir: str):
    """
    Create YOLO format dataset from Kaggle signature data
    """
    
    # Create output directory structure
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # Load CSV files
    train_df = pd.read_csv(f"{input_dir}/Train_data.csv", 
                          names=['filename', 'label', 'x1', 'y1', 'x2', 'y2'])
    test_df = pd.read_csv(f"{input_dir}/Test_data.csv", 
                         names=['filename', 'label', 'x1', 'y1', 'x2', 'y2'])
    
    print(f"Found {len(train_df)} training images and {len(test_df)} test images")

    convert_to_yolo(input_dir, train_df, output_dir, "train")
    convert_to_yolo(input_dir, test_df, output_dir, "val")
    
    create_yolo_dataset_yaml(output_dir)
    
    print(f"Dataset converted successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(test_df)}")
    print(f"Dataset config: {output_dir}/dataset.yaml")

if __name__ == "__main__":
    input_dir = "kagle_archive/images"
    output_dir = "yolo_signature_dataset"

    create_yolo_dataset(input_dir, output_dir)

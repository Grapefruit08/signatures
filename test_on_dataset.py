"""
Simple batch signature detection script
"""
import glob
import os
from pathlib import Path
from cli_detect import detect_signatures


def main():
    # Paths
    dataset_path = "dataset/dataset"
    output_dir = "dataset_onnx_output"
    model_path = "runs/detect/signature_detection_old/yolov8n_signatures_augmented/weights/best.onnx"
    
    # Create output directories
    os.makedirs(f"{output_dir}/signatures", exist_ok=True)
    os.makedirs(f"{output_dir}/background", exist_ok=True)
    
    print("Processing signatures...")
    # Process signatures folder
    signature_files = glob.glob(f"{dataset_path}/signatures/*.jpg")
    for i, image_path in enumerate(signature_files):
        print(f"  {i+1}/{len(signature_files)}: {Path(image_path).name}")
        detect_signatures(
            image_path=image_path,
            output_dir=f"{output_dir}/signatures",
            model_path=model_path,
            confidence=0.25,
            visualize=True
        )
    
    print("Processing background...")
    # Process background folder  
    background_files = glob.glob(f"{dataset_path}/background/*.jpg")
    for i, image_path in enumerate(background_files):
        print(f"  {i+1}/{len(background_files)}: {Path(image_path).name}")
        detect_signatures(
            image_path=image_path,
            output_dir=f"{output_dir}/background", 
            model_path=model_path,
            confidence=0.25,
            visualize=True
        )
    
    print(f"Done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
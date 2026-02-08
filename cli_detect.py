"""
Signature Detection CLI Tool using Click
"""
import click
import os
import sys
import json
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


def write_detection_results(detections: YOLO, image_path: str, output_dir: str, image_size: tuple[int, int]):
    """
    Write detection results to a JSON file

    Args:
        detections (YOLO): YOLO detection results object
        image_path (str): Path to the input image
        output_dir (str): Directory to save the JSON results
        image_size (tuple[int, int]): (width, height) of the input image
    """

    img_width, img_height = image_size
    
    # Generate JSON output path based on image name
    image_name = Path(image_path).stem
    json_output_path = os.path.join(output_dir, f"{image_name}_detections.json")

    # Write detection results to JSON file
    detection_results = {
        "image_name": os.path.basename(image_path),
        "image_size": {
            "width": img_width,
            "height": img_height
        },
        "detections": []
    }
        
    num_detections = 0
    if detections.boxes is not None and len(detections.boxes) > 0:
        boxes = detections.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = detections.boxes.conf.cpu().numpy()
            
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
                
            detection_data = {
                "confidence": float(conf),
                "bbox": {
                    "x_min": float(x1),
                    "y_min": float(y1), 
                    "x_max": float(x2),
                    "y_max": float(y2)
                }
            }
                
            detection_results["detections"].append(detection_data)
            num_detections += 1
                
            click.echo(f"   Detection {i+1}: confidence={conf:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    
    # Write JSON to file
    with open(json_output_path, 'w') as f:
        json.dump(detection_results, f, indent=2)

    click.echo(f"Results saved to: {json_output_path}")
    click.echo(f"Found {num_detections} signatures")


def detect_signatures(image_path: str, output_dir: str, model_path: str, visualize: bool = False, confidence: float = 0.25) -> bool:
    """
    Perform signature detection on an image and save results
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save detection results
        model_path (str): Path to ONNX model
        visualize (bool): Whether to save annotated image
        confidence (float): Detection confidence threshold
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Check if input image exists
    if not os.path.exists(image_path):
        click.echo(f"Error: Input image not found: {image_path}", err=True)
        return False
    
    # Check if model exists
    if not os.path.exists(model_path):
        click.echo(f"Error: ONNX model not found: {model_path}", err=True)
        click.echo("Please ensure the model exists or run training first.", err=True)
        return False
    
    # load the model
    try:
        click.echo(f"Loading ONNX model: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        return False
    
    # Load image to get dimensions for output
    try:
        with Image.open(image_path) as image:
            img_width, img_height = image.size
    except Exception as e:
        click.echo(f"Error opening image: {e}", err=True)
        return False
    
    # Run inference
    try:
        click.echo(f"Processing image: {image_path}")
        results = model(image_path, save=False, conf=confidence, verbose=False)
        detections = results[0]
    except Exception as e:
        click.echo(f"Error during inference: {e}", err=True)
        return False
    
    # Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write results to json
    try:
        write_detection_results(detections, image_path, output_dir, (img_width, img_height))
    except Exception as e:
        click.echo(f"Error writing results: {e}", err=True)
        return False
    
    # Save visualized image if requested
    if visualize:
        try:
            # Generate visualization output path based on image name
            image_name = Path(image_path).stem
            vis_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
            
            # Get annotated image from YOLO
            annotated_img = detections.plot()
            
            # YOLO returns BGR, PIL uses RGB ?
            annotated_img_rgb = annotated_img[:, :, ::-1]  # BGR to RGB
            
            # Save the annotated image using PIL
            pil_image = Image.fromarray(annotated_img_rgb)
            pil_image.save(vis_path)
            click.echo(f"Visualization saved to: {vis_path}")

        except Exception as e:
            click.echo(f"Error in visualization: {e}", err=True)
            return False
        
    return True


@click.command()
@click.option(
    "--image_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input image file.",
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(),
    required=True,
    help="Directory to save detection results (JSON and optional visualization).",
)
@click.option(
    "--model_path",
    "-m",
    type=click.Path(exists=True),
    default="best.onnx",
    help="Path to the ONNX model file (default: best.onnx).",
)
@click.option(
    "--confidence",
    "-c",
    type=float,
    default=0.25,
    help="Detection confidence threshold (default: 0.25).",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Save annotated image with bounding boxes drawn.",
)
def detect(
    image_path: str,
    output_dir: str,
    model_path: str = "best.onnx",
    confidence: float = 0.25,
    visualize: bool = False,
) -> None:
    """
    Perform signature detection on an image using ONNX model.
    
    Detects signatures in images
    Outputs detection results to a JSON file in output directory
    JSON includes:
        - image name
        - image size
        - list of detections with confidence
        - bounding box coordinates
    Optionally saves a visualization of drawn bbox
    
    Example:
        Example CLI usage:
        python cli_detect.py 
        -i "dataset/dataset/signatures/9.jpg" 
        -o "results" 
        -m "model_zoo/yolo_non_augmented/best.onnx" 
        -c 0.25 --visualize
    """ 
   
    # Run detection
    success = detect_signatures(
        image_path=image_path,
        output_dir=output_dir,
        model_path=model_path,
        visualize=visualize,
        confidence=confidence
    )
    
    if success:
        click.echo("\nDetection completed successfully!")
        sys.exit(0)
    else:
        click.echo("\nDetection failed!", err=True)
        sys.exit(1)


if __name__ == "__main__":
    detect()

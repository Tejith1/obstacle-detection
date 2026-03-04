"""
Quick script to check what classes your best.pt model was trained on.
Run this to see what each class number means for YOUR specific model.
"""
import torch
import sys
import os
import pathlib

# Fix for Windows when loading Linux-trained YOLO weights
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Add yolov5 to path for loading the model
sys.path.insert(0, 'yolov5')

def check_model_classes(model_path='best.pt'):
    """Load model and print all class names with their IDs."""
    try:
        print(f"Loading model from: {model_path}")
        print("-" * 50)
        
        # Load with weights_only=False for YOLOv5 models (trusted source)
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Try to get class names from different possible locations
        if 'model' in model:
            if hasattr(model['model'], 'names'):
                names = model['model'].names
            elif 'names' in model:
                names = model['names']
            else:
                print("Could not find class names in model structure")
                return
        elif 'names' in model:
            names = model['names']
        else:
            print("Model structure not recognized. Trying alternative methods...")
            # Try loading with YOLOv5 loader
            sys.path.insert(0, 'yolov5')
            from models.common import DetectMultiBackend
            from utils.torch_utils import select_device
            device = select_device('')
            model_loaded = DetectMultiBackend(model_path, device=device)
            names = model_loaded.names
        
        print(f"\n✅ Your model has {len(names)} classes:\n")
        print("Class ID | Class Name")
        print("-" * 30)
        
        if isinstance(names, dict):
            for class_id, class_name in names.items():
                print(f"{class_id:8} | {class_name}")
        elif isinstance(names, list):
            for class_id, class_name in enumerate(names):
                print(f"{class_id:8} | {class_name}")
        else:
            print(f"Names type: {type(names)}")
            print(names)
        
        print("\n" + "-" * 50)
        print("These are the classes your model can detect!")
        print("When you see 'class 9', look up ID 9 in the table above.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {model_path}")
        print("Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    import os
    
    # Check if best.pt exists
    if not os.path.exists('best.pt'):
        print("❌ best.pt not found in current directory!")
        print("Please run this script from the project root folder.")
    else:
        check_model_classes('best.pt')

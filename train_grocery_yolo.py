import os
import pandas as pd
from ultralytics import YOLO

# === Path Definitions === 
base_dir = r'C:\Users\Carl\Downloads\Groceries_project\GroceryStoreDataset\dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
classes_csv = os.path.join(base_dir, 'classes.csv')
yaml_path = os.path.join(base_dir, 'data.yaml')

# === data.yaml from classes.csv ===
df = pd.read_csv(classes_csv)
class_names = df['Class Name (str)'].tolist()

data_yaml_content = {
    'train': train_dir,
    'val': val_dir,
    'nc': len(class_names),
    'names': class_names
}

with open(yaml_path, 'w') as f: 
    f.write(f"train: {train_dir}\n")
    f.write(f"val: {val_dir}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write(f"names: {class_names}\n")

print(f"data.yaml generated at {yaml_path}")

# === Train Yolov8 ===
model = YOLO('yolov8n.pt')

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    project='runs',
    name='grocery_yolo'
)

# === Model export to ONNX ===
export_path = model.export(format='onnx')
print(f"Model exported to ONNX: {export_path}")

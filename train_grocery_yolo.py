# import os
# import pandas as pd
# from ultralytics import YOLO

# # === Path Definitions === 
# base_dir = r'C:\Users\Carl\Downloads\Groceries_project\GroceryStoreDataset\dataset'
# train_dir = os.path.join(base_dir, 'train')
# val_dir = os.path.join(base_dir, 'val')
# classes_csv = os.path.join(base_dir, 'classes.csv')
# yaml_path = os.path.join(base_dir, 'data.yaml')

# # === data.yaml from classes.csv ===
# df = pd.read_csv(classes_csv)
# class_names = df['Class Name (str)'].tolist()

# data_yaml_content = {
#     'train': train_dir,
#     'val': val_dir,
#     'nc': len(class_names),
#     'names': class_names
# }

# with open(yaml_path, 'w') as f: 
#     f.write(f"train: {train_dir}\n")
#     f.write(f"val: {val_dir}\n")
#     f.write(f"nc: {len(class_names)}\n")
#     f.write(f"names: {class_names}\n")

# print(f"data.yaml generated at {yaml_path}")

# # === Train Yolov8 ===
# model = YOLO('yolov8n.pt')

# model.train(
#     data=yaml_path,
#     epochs=50,
#     imgsz=640,
#     project='runs',
#     name='grocery_yolo'
# )

# # === Model export to ONNX ===
# export_path = model.export(format='onnx')
# print(f"Model exported to ONNX: {export_path}")

import os
from ultralytics import YOLO
import yaml

# === Path Definitions === 
base_dir = r'C:\Users\Carl\Downloads\Groceries_project\GroceryStoreDataset\dataset'
train_dir = os.path.join(base_dir, 'train/images')
val_dir = os.path.join(base_dir, 'valid/images')
test_dir = os.path.join(base_dir, 'test/images')
yaml_path = os.path.join(base_dir, 'data.yaml')

# === Define class mapping ===
class_names = {
    0: "-",
    1: "Apple",
    2: "Asparagus",
    3: "Avocado",
    4: "Banana",
    5: "Beans",
    6: "Blackberries",
    7: "Blueberries",
    8: "Book",
    9: "Broccoli",
    10: "Brussel Sprouts",
    11: "Butter",
    12: "Cabbage",
    13: "Cantaloupe",
    14: "Carrots",
    15: "Cauliflower",
    16: "Cerealbox",
    17: "Cheese",
    18: "Clementine",
    19: "Coffee",
    20: "Corn",
    21: "Cucumber",
    22: "Detergent",
    23: "Drinks",
    24: "Egg",
    25: "Eggplant",
    26: "Eggs",
    27: "Galia",
    28: "Grapes",
    29: "Honeydew",
    30: "Juice",
    31: "Lettuce",
    32: "Meat",
    33: "Milk",
    34: "Mushrooms",
    35: "Nectarine",
    36: "Orange",
    37: "Oranges",
    38: "Pineapple",
    39: "Plum",
    40: "Pomegranate",
    41: "Raspberries",
    42: "Salad",
    43: "Sauce",
    44: "Spinach",
    45: "Squash",
    46: "Strawberries",
    47: "Strawberry",
    48: "Tofu",
    49: "Tomatoes",
    50: "Watermelon",
    51: "Yogurt",
    52: "Zucchini",
    53: "beverage",
    54: "food-box",
    55: "fruit",
    56: "utility-box",
    57: "vegetable"
}

# === Compose YAML content ===
data_yaml = {
    'train': train_dir.replace("\\", "/"),
    'val': val_dir.replace("\\", "/"),
    'test': test_dir.replace("\\", "/"),
    'names': class_names
}

# === Save YAML to file ===
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✅ data.yaml generated at: {yaml_path}")

# === Train YOLOv8 ===
model = YOLO('yolov8n.pt')  # Replace with yolov8s.pt, yolov8m.pt, etc., if needed

model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    project='runs',
    name='grocery_yolo'
)

# === Export model to ONNX ===
export_path = model.export(format='onnx')
print(f"✅ Model exported to ONNX: {export_path}")

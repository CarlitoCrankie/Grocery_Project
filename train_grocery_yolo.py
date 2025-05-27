import os
from ultralytics import YOLO
import yaml

# === Path Definitions === 
base_dir = os.path.dirname(os.path.abspath(__file__))

# Then construct paths relative to that
dataset_dir = os.path.join(base_dir, 'dataset')
train_dir = os.path.join(dataset_dir, 'train', 'images')
val_dir = os.path.join(dataset_dir, 'valid', 'images')
test_dir = os.path.join(dataset_dir, 'test', 'images')
yaml_path = os.path.join(base_dir, 'dataset', 'data.yaml')

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

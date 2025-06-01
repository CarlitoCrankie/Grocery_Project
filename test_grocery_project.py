import os
from ultralytics import YOLO
import cv2

# === Load the trained model ===
model_path = r'C:\Users\Carl\Downloads\Groceries_project\GroceryStoreDataset\dataset\best.pt'  # Update if you used a different path
model = YOLO(model_path)

# === Source image path ===
source_image_path = 'dataset/360_F_65706597_uNm2SwlPIuNUDuMwo6stBd81e25Y8K8s.jpg'

# === Run prediction ===
results = model.predict(source=source_image_path, save=True)

# === Find latest prediction folder ===
detect_dir = 'runs/detect'
latest_folder = max([os.path.join(detect_dir, d) for d in os.listdir(detect_dir)], key=os.path.getmtime)

# === Get predicted image path ===
image_filename = os.path.basename(source_image_path)
pred_image_path = os.path.join(latest_folder, image_filename)

# === Load and display image ===
img = cv2.imread(pred_image_path)
cv2.imshow("Predicted", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

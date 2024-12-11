import os
import json

from tqdm import tqdm
from ultralytics import YOLO

from config import YOLO_MODEL_PATH, CLASSES, OUTPUT_FOLDER, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
from load_dataset import custom_dataloader


def main():
    detection_model = YOLO(YOLO_MODEL_PATH)

    YOLO_OUTPUT_FILEPATH = os.path.join(OUTPUT_FOLDER, "results.json")

    # Define custom classes
    if CLASSES:
        detection_model.set_classes(CLASSES)

    # train_loader = custom_dataloader("train")
    # val_loader = custom_dataloader("val")
    test_loader = custom_dataloader("test")

    data = dict()
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        if batch_idx > 10:
            break
        results = detection_model.predict(batch["images"], conf=0.25)
        for index, result in enumerate(results):
            image_data = []
            for box in result.boxes:
                image_data.append({
                    "bbox": box.xyxyn.tolist(),
                    "category_id": box.cls.item(),
                    "confidence": box.conf.item(),
                    "category_name": detection_model.names.get(box.cls.item())
                })
            data[batch["ids"][index]] = image_data
            result.save(os.path.join(YOLO_OUTPUT_FILEPATH, f"{batch_idx * TEST_BATCH_SIZE + index}.jpg"))

    with open(YOLO_OUTPUT_FILEPATH, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Saved to {YOLO_OUTPUT_FILEPATH}")

if __name__ == "__main__":
    main()
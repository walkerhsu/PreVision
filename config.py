YOLO_MODEL_PATH = "weights/yolov8s-worldv2.pt"
CLASSES = [
    "car", "truck", "bus", "motorcycle", "bicycle", "tricycle", "van", "suv", "trailer", 
    "construction vehicle", "moped", "recreational vehicle", "pedestrian", "cyclist", 
    "wheelchair", "stroller", "traffic light", "traffic sign", "traffic cone", 
    "traffic island", "traffic box", "barrier", "bollard", "warning sign", "debris", 
    "machinery", "dustbin", "concrete block", "cart", "chair", "basket", "suitcase", 
    "dog", "phone booth"
]
OUTPUT_FOLDER = "results"
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
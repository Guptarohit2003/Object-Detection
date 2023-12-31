import cv2
import cvzone
from ultralytics import YOLO
import math
from sort import *


cap = cv2.VideoCapture("C:/PROFO/object detection/Videos/cars.mp4")

model = YOLO("../YOLO-weights/yolov8n.pt")

classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

mask = cv2.imread(
    "C:\PROFO\object detection\Projects\Object-Detection\Car Counter\mask.png"
)


while True:
    success, img = cap.read()
    imgregion = cv2.bitwise_and(img, mask)
    results = model(imgregion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            w, h = x2 - x1, y2 - y1

            # confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass in ["car", "truck", "motorbike", "bus"] and conf > 0.3:
                cvzone.putTextRect(
                    img,
                    f"{currentclass} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=0.6,
                    thickness=1,
                    offset=3,
                )
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=3)

    cv2.imshow("Image", img)
    # cv2.imshow("Image REgion", imgregion)
    cv2.waitKey(0)

import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/yolo_webcam_train/weights/best.pt"
CONF_THRESHOLD = 0.25

def main():
    model = YOLO(MODEL_PATH)
    class_names = model.names

    print("Loaded classes:", class_names)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam tidak bisa dibuka")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        annotated = results[0].plot(
            conf=True,
            labels=True
        )

        counts = {}
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]
                counts[cls_name] = counts.get(cls_name, 0) + 1

        y = 30
        for name, total in counts.items():
            cv2.putText(
                annotated,
                f"{name} : {total}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y += 30

        cv2.imshow("YOLOv8 Webcam Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

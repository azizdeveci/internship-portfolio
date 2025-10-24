# import libraries

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model

model = YOLO('/home/huawei/Documents/internship/Tasks/yolo_project/models/yolov8n.pt')


#opening the video file

cap=cv2.VideoCapture('/home/huawei/Documents/internship/Tasks/yolo_project/videos/2.mp4')

success, frame = cap.read()
if not success:
    print("Failed to read video")
    exit()

# resizing the frame

frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
frame_height, frame_width = frame.shape[:2]

line_x = int(frame_width * 0.5)
offset = 10  # Allowable error margin for line crossing

giren = 0
cikan = 0
counted_ids=set()
person_last_x = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)

    result = model.track(frame, persist=True, stream=False, conf=0.05, iou=0.05, tracker="bytetrack.yaml" )

    if result[0].boxes.id is not None:
        ids = result[0].boxes.id.int().tolist()
        classes = result[0].boxes.cls.int().tolist()
        xyxy = result[0].boxes.xyxy

        for i,box in enumerate(xyxy):
             cls_id = classes[i]
             track_id= ids[i]
             class_name = model.names[cls_id]

             if class_name != 'person':
                 continue

             x1, y1, x2, y2 = map(int, box)
             cx=int((x1 + x2) / 2)
             cy=int((y1 + y2) / 2)

             previous_x = person_last_x.get(track_id, None)
             person_last_x[track_id] = cx

             if previous_x is not None:
                 #sagdan sola gecenler
                 if previous_x > line_x >=cx:
                     if track_id not in counted_ids:
                         cikan += 1
                         counted_ids.add(track_id)

                # soldan saga gecenler
                 elif  previous_x < line_x <= cx:
                     if track_id not in counted_ids:
                         giren += 1
                         counted_ids.add(track_id)

            # Draw bounding box and ID
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
             cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    #dikeyden ayırma çizgisi
    cv2.line(frame, (line_x, 0), (line_x, frame_height), (0, 0, 255), 2)

    # Sayıları ekrana yazdırma
    cv2.putText(frame, f'Giris ==>: {giren}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'<== Cikis: {cikan}', (580, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Kisi Sayma", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print("Completed ...")
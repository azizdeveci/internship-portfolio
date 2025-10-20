import cv2
import numpy as np
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def load_yolo_tiny():
    # YOLOv3-tiny modelini yükle
    net = cv2.dnn.readNet("models/yolov3-tiny.weights", "models/yolov3-tiny.cfg")
    
    with open("models/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def tiny_yolo_detection(image_path):
    net, classes, output_layers = load_yolo_tiny()
    
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    if img is None:
        print("Görüntü yüklenemedi!")
        return None, 0
        
    height, width = img.shape[:2]
    
    # Blob oluştur
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Tespitleri işle
    boxes, confs, class_ids = [], [], []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confs.append(float(confidence))
                class_ids.append(class_id)
    
    # NMS uygula
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    
    # Sonuçları çiz
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    detected_objects = {}
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confs[i], 2))
            color = colors[class_ids[i]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence}", (x, y + 20), font, 1, color, 2)
            
            # Tespit edilen nesneleri say
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1
    
    cv2.imshow("YOLOv3-tiny Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # İstatistikleri yazdır
    print(f"\n--- YOLOv3-tiny Tespit Sonuçları ---")
    print(f"Toplam tespit edilen nesne: {len(indexes)}")
    for obj, count in detected_objects.items():
        print(f"{obj}: {count} adet")
    
    return img, len(indexes), detected_objects

# Kullanım
result_image, detected_count, objects_info = tiny_yolo_detection("images/human.jpeg")

# Klasörü oluştur
os.makedirs('result_images', exist_ok=True)

# Sonuç resmini kaydet
if result_image is not None:
    output_path = os.path.join('result_images', 'yolov3_results.jpg')
    cv2.imwrite(output_path, result_image)
    print(f"Sonuç resmi kaydedildi: {output_path}")
else:
    print("Sonuç resmi kaydedilemedi!")

print(f"Tespit edilen nesne sayısı: {detected_count}")
print(f"Detaylı bilgi: {objects_info}")
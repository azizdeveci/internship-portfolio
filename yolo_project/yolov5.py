import cv2
import numpy as np
import os
from ultralytics import YOLO

os.environ['QT_QPA_PLATFORM'] = 'xcb'

def ultralytics_yolov5_detection(image_path, model_path="models/yolov5nu.pt"):
    
    
    # Modeli yükle
    if not os.path.exists(model_path):
        print(f"Model dosyası bulunamadı: {model_path}")
        return None, 0, {}
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None, 0, {}
    
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    if img is None:
        print("Görüntü yüklenemedi!")
        return None, 0, {}
    
    # Inference
    results = model(img)
    
    # Sonuçları işle
    result = results[0]
    detected_objects = {}
    
    # Tespitleri çiz
    for box in result.boxes:
        confidence = box.conf.item()
        
        if confidence > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls.item())
            label = model.names[class_id]
            
            # Bounding box çiz
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Nesneleri say
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1
    
    # Görselleştir
    cv2.imshow("YOLOv5 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # İstatistikler
    high_conf_boxes = [box for box in result.boxes if box.conf.item() > 0.5]
    print(f"\n--- YOLOv5 Tespit Sonuçları ---")
    print(f"Toplam tespit edilen nesne: {len(high_conf_boxes)}")
    
    for obj, count in detected_objects.items():
        print(f"{obj}: {count} adet")
    
    return img, len(high_conf_boxes), detected_objects

# Kullanım
if __name__ == "__main__":
    result_image, count, info = ultralytics_yolov5_detection("images/human.jpeg")
    
    if result_image is not None:
        os.makedirs('result_images', exist_ok=True)
        cv2.imwrite('result_images/yolov5_ultralytics_result.jpg', result_image)
        print("Sonuç kaydedildi!")
from ultralytics import YOLO
import cv2
import os

# Model yükle
model = YOLO('yolov8n-seg.pt')

# Tahmin yap
results = model('/home/huawei/Documents/internship/Tasks/human.jpeg')


# Sonuçları kaydet
for i, result in enumerate(results):
    # Built-in plot fonksiyonu ile görselleştir ve kaydet
    result_image = result.plot()  # Mask ve box'ları içeren resim
    
    # RGB'den BGR'ye çevir (OpenCV formatı)
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    # Kaydet
    output_path = os.path.join('images', f'segmentation_result_{i}.jpg')
    cv2.imwrite(output_path, result_image_bgr)
    print(f"Mask'lı resim kaydedildi: segmentation_result_{i}.jpg")
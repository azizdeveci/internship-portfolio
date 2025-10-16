from ultralytics import YOLO
import time

def test_model(model_path, model_type, image_path):
    """Model test et ve sonuçları göster"""
    print(f"\n{'='*50}")
    print(f"🤖 {model_type.upper()} MODELİ TEST EDİLİYOR")
    print(f"Model: {model_path}")
    print(f"{'='*50}")
    
    # Model yükle
    model = YOLO(model_path)
    
    # Süre ölç
    start_time = time.time()
    results = model(image_path)
    end_time = time.time()
    
    print(f"⏱️ İşlem süresi: {end_time - start_time:.2f} saniye")
    
    # Sonuçları göster
    for result in results:
        if model_type == "detection":
            print(f"🎯 Tespit edilen nesne sayısı: {len(result.boxes)}")
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"  {i+1}. {class_name}: {confidence:.2f}")
                    
        elif model_type == "classification":
            print(f"📊 En yüksek olasılık: {result.probs.top1}")
            print(f"Güven skoru: {result.probs.top1conf:.2f}")
            print(f"🏆 En iyi 3 sınıf:")
            for i, (class_name, confidence) in enumerate(result.probs.top3, 1):
                print(f"  {i}. {class_name}: {confidence:.2f}")
                
        elif model_type == "segmentation":
            print(f"🎯 Tespit edilen nesne sayısı: {len(result.boxes)}")
            print(f"✂️ Bölütleme maskesi sayısı: {len(result.masks) if result.masks is not None else 0}")

# Test görüntüsü
image_path = '/home/huawei/Documents/internship/Tasks/bus.jpg'

# Tüm modelleri test et
models = [
    ('yolov8n.pt', 'detection'),
    ('yolov8n-cls.pt', 'classification'),
    ('yolov8n-seg.pt', 'segmentation')
]

for model_path, model_type in models:
    try:
        test_model(model_path, model_type, image_path)
    except Exception as e:
        print(f"❌ Hata: {model_path} - {str(e)}")

print(f"\n{'='*50}")
print("✅ TÜM TESTLER TAMAMLANDI")
print(f"{'='*50}")

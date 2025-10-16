from ultralytics import YOLO
import time

def test_yolo_version(model_name, task_type="detect"):
    """YOLO versiyonunu test et"""
    print(f"\n{'='*60}")
    print(f"🤖 {model_name.upper()} TEST EDİLİYOR")
    print(f"{'='*60}")
    
    try:
        # Model yükle
        start_load = time.time()
        model = YOLO(model_name)
        load_time = time.time() - start_load
        
        print(f"✅ Model başarıyla yüklendi")
        print(f"⏱️ Yükleme süresi: {load_time:.2f} saniye")
        print(f"📊 Model bilgileri:")
        print(f"   - Görev türü: {model.task}")
        print(f"   - Sınıf sayısı: {len(model.names)}")
        print(f"   - Model boyutu: {model.model_size_mb:.1f} MB" if hasattr(model, 'model_size_mb') else "   - Model boyutu: Bilinmiyor")
        
        # Test görüntüsü ile test et
        test_image = '/home/huawei/Documents/internship/Tasks/bus.jpg'
        
        start_inference = time.time()
        results = model(test_image)
        inference_time = time.time() - start_inference
        
        print(f"⚡ Çıkarım süresi: {inference_time:.2f} saniye")
        
        # Sonuçları göster
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                print(f"🎯 Tespit edilen nesne sayısı: {len(result.boxes)}")
            elif hasattr(result, 'probs'):
                print(f"📊 Sınıflandırma: {result.probs.top1}")
            elif hasattr(result, 'masks') and result.masks is not None:
                print(f"✂️ Bölütleme maskesi: {len(result.masks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return False

# Desteklenen YOLO versiyonları
yolo_models = [
    # YOLOv8 - Detection
    "yolov8n.pt",      # Nano
    "yolov8s.pt",      # Small
    "yolov8m.pt",      # Medium
    "yolov8l.pt",      # Large
    "yolov8x.pt",      # Extra Large
    
    # YOLOv8 - Classification
    "yolov8n-cls.pt",
    "yolov8s-cls.pt",
    "yolov8m-cls.pt",
    "yolov8l-cls.pt",
    "yolov8x-cls.pt",
    
    # YOLOv8 - Segmentation
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
    
    # YOLOv8 - Pose (Pose estimation)
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
    
    # YOLOv8 - OBB (Oriented Bounding Box)
    "yolov8n-obb.pt",
    "yolov8s-obb.pt",
    "yolov8m-obb.pt",
    "yolov8l-obb.pt",
    "yolov8x-obb.pt",
    
    # YOLOv9
    "yolov9t.pt",      # Tiny
    "yolov9s.pt",      # Small
    "yolov9m.pt",      # Medium
    "yolov9c.pt",      # Custom
    "yolov9e.pt",      # Extra
    
    # YOLOv10
    "yolov10n.pt",     # Nano
    "yolov10s.pt",     # Small
    "yolov10m.pt",     # Medium
    "yolov10b.pt",     # Base
    "yolov10l.pt",     # Large
    "yolov10x.pt",     # Extra Large
    
    # YOLOv11
    "yolov11n.pt",     # Nano
    "yolov11s.pt",     # Small
    "yolov11m.pt",     # Medium
    "yolov11l.pt",     # Large
    "yolov11x.pt",     # Extra Large
]

print("🚀 YOLO VERSİYONLARI TEST EDİLİYOR")
print("=" * 60)

successful_models = []
failed_models = []

# İlk 5 modeli test et (hızlı test için)
test_models = yolo_models[:5]

for model_name in test_models:
    if test_yolo_version(model_name):
        successful_models.append(model_name)
    else:
        failed_models.append(model_name)

print(f"\n{'='*60}")
print("📊 TEST SONUÇLARI")
print(f"{'='*60}")
print(f"✅ Başarılı: {len(successful_models)}")
print(f"❌ Başarısız: {len(failed_models)}")

if successful_models:
    print(f"\n🎉 Başarılı modeller:")
    for model in successful_models:
        print(f"   - {model}")

if failed_models:
    print(f"\n⚠️ Başarısız modeller:")
    for model in failed_models:
        print(f"   - {model}")

print(f"\n💡 Tüm modelleri test etmek için:")
print(f"   test_models = yolo_models  # Tüm modelleri test et")
print(f"   # VEYA")
print(f"   test_models = yolo_models[5:10]  # Belirli aralığı test et")


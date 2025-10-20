import cv2
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import sys

sys.path.append(os.path.join(os.getcwd(), "YOLOv6"))
from yolov6.core.inferer import Inferer


os.environ['QT_QPA_PLATFORM'] = 'xcb'

def load_yolov6_official(model_path="models/yolov6n.pt"):
    """Resmi YOLOv6 implementasyonu ile model yükle"""
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return None
    
    try:
        print("🔄 YOLOv6 modeli yükleniyor...")
        
        # YOLOv6 modüllerini import et
        from yolov6.core.inferer import Inferer
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Inferer oluştur
        inferer = Inferer(source='', weights=model_path, device=device, half=False)
        
        print(f"✅ YOLOv6 modeli başarıyla yüklendi")
        print(f"📍 Cihaz: {device}")
        
        return inferer
        
    except ImportError as e:
        print(f"❌ YOLOv6 modülleri bulunamadı: {e}")
        print("📦 Lütfen YOLOv6'ı kurun: pip install yolov6")
        return None
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}")
        return None

def yolov6_detection(image_path, inferer):
    """YOLOv6 ile nesne tespiti"""
    
    if not os.path.exists(image_path):
        print(f"❌ Görüntü dosyası bulunamadı: {image_path}")
        return None, 0, {}
    
    try:
        # Inference yap
        print("🔍 Nesne tespiti yapılıyor...")
        detections, img = inferer.infer(
            image_path, 
            conf_thres=0.5, 
            iou_thres=0.45, 
            image_size=640,
            save_dir=None
        )
        
        # OpenCV formatına çevir
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        detected_objects = {}
        
        # Tespitleri işle
        if detections is not None and len(detections) > 0:
            for det in detections:
                if len(det) >= 6:  # [x1, y1, x2, y2, conf, class]
                    x1, y1, x2, y2, conf, class_id = det[:6]
                    
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_id = int(class_id)
                    
                    # COCO sınıf isimleri
                    coco_classes = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                        'toothbrush'
                    ]
                    
                    if class_id < len(coco_classes):
                        label = coco_classes[class_id]
                        
                        # Bounding box çiz
                        color = (0, 255, 0)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Nesneleri say
                        if label in detected_objects:
                            detected_objects[label] += 1
                        else:
                            detected_objects[label] = 1
        
        # Görselleştir
        cv2.imshow("YOLOv6 Detection", img)
        print("👀 Sonuçlar gösteriliyor... (Pencereyi kapatmak için bir tuşa basın)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # İstatistikler
        total_detections = len(detections) if detections is not None else 0
        print(f"\n📊 YOLOv6 Tespit Sonuçları")
        print(f"{'='*40}")
        print(f"🔍 Toplam tespit edilen nesne: {total_detections}")
        
        if detected_objects:
            for obj, count in detected_objects.items():
                print(f"   {obj}: {count} adet")
        else:
            print("   🤷 Hiç nesne tespit edilemedi")
        
        return img, total_detections, detected_objects
        
    except Exception as e:
        print(f"❌ Tespit sırasında hata: {e}")
        return None, 0, {}

# Ana program
if __name__ == "__main__":
    # Model ve görüntü yolları
    model_path = "models/yolov6n.pt"
    image_path = "images/human.jpeg"
    
    # Dosya kontrolleri
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("📥 Lütfen model dosyasını indirin:")
        print("wget -O models/yolov6n.pt https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt")
        exit(1)
    
    if not os.path.exists(image_path):
        print(f"❌ Görüntü dosyası bulunamadı: {image_path}")
        exit(1)
    
    # Modeli yükle
    inferer = load_yolov6_official(model_path)
    
    if inferer is None:
        print("❌ Model yüklenemedi!")
        exit(1)
    
    # Nesne tespiti yap
    result_image, detected_count, objects_info = yolov6_detection(image_path, inferer)
    
    # Sonuçları kaydet
    os.makedirs('result_images', exist_ok=True)
    
    if result_image is not None:
        output_path = os.path.join('result_images', 'yolov6_results.jpg')
        cv2.imwrite(output_path, result_image)
        print(f"💾 Sonuç resmi kaydedildi: {output_path}")
        print(f"✅ YOLOv6 başarıyla çalıştı!")
    else:
        print("❌ Sonuç resmi kaydedilemedi!")
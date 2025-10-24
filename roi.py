import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Global Değişkenler ---
drawing = False  # Farenin basılı olup olmadığını takip eder
ix, iy = -1, -1  # Dikdörtgenin başlangıç koordinatları
rect = (0, 0, 0, 0) # (x, y, w, h) formatında ROI dikdörtgeni
img_copy = None # Görüntünün kopyası üzerine çizim yapmak için

def draw_rectangle(event, x, y, flags, param):
    """Farenin hareketlerini takip eden ve dikdörtgen çizen fonksiyon."""
    global ix, iy, drawing, rect, img_copy
    
    # Farenin sol tuşuna basıldığında
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        print(f"Başlangıç: ({ix}, {iy})")

    # Fare hareket ettiğinde (ve sol tuş basılıysa)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # Önceki çizimi temizlemek için görüntünün taze kopyasını al
            img_copy = img.copy() 
            # O anki dikdörtgeni çiz
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    # Farenin sol tuşu bırakıldığında
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Son dikdörtgeni çiz
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        # Dikdörtgenin (x, y, w, h) formatında koordinatlarını kaydet
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        print(f"Bitiş: ({x}, {y})")
        print(f"Kaydedilen Dikdörtgen (rect): {rect}")

def run_grabcut():
    """Kullanıcının seçtiği dikdörtgeni kullanarak GrabCut algoritmasını çalıştırır."""
    global img, rect
    
    print("\nGrabCut algoritması çalıştırılıyor...")
    
    # GrabCut'ın ihtiyaç duyduğu maske ve modelleri hazırla
    # Maske, görüntünün her pikselinin durumunu (arka plan, ön plan, vb.) tutar
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Arka plan ve ön plan modelleri için geçici diziler
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut algoritmasını çalıştır
    # cv2.GC_INIT_WITH_RECT modu, bir dikdörtgen ile başladığımızı belirtir
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        print("GrabCut tamamlandı.")
    except Exception as e:
        print(f"Hata: {e}. Lütfen geçerli bir dikdörtgen çizdiğinizden emin olun.")
        return

    # GrabCut'tan sonra maske, 0 (kesin arka plan), 1 (kesin ön plan), 
    # 2 (muhtemel arka plan), 3 (muhtemel ön plan) değerlerini içerir.
    
    # Biz 0 ve 2 olanları (tüm arka planları) 0'a,
    # 1 ve 3 olanları (tüm ön planları) 1'e ayarlıyoruz.
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Orijinal görüntüyü bu maske ile çarparak arka planı siyah yaparız
    # result_bgr = img * mask2[:, :, np.newaxis]

    # --- Şeffaf Arka Plan (PNG) Oluşturma ---
    
    # Orijinal BGR görüntüyü B, G, R kanallarına ayır
    b, g, r = cv2.split(img)
    
    # Yeni alpha (şeffaflık) kanalı oluştur. 
    # Maskenin 0 olduğu yerler 0 (şeffaf), 1 olduğu yerler 255 (opak) olacak.
    alpha = np.where(mask2 == 0, 0, 255).astype('uint8')

    # B, G, R ve Alpha kanallarını birleştirerek BGRA (4 kanallı) görüntü oluştur
    result_rgba = cv2.merge((b, g, r, alpha))

    # Sonucu kaydet
    output_filename = 'sonuc_arka_plan_silindi.png'
    cv2.imwrite(output_filename, result_rgba)
    print(f"Sonuç '{output_filename}' olarak kaydedildi.")

    # Sonucu Matplotlib ile göster (OpenCV'nin imshow'u şeffaflığı iyi göstermez)
    plt.figure(figsize=(10, 7))
    # Matplotlib RGB formatında bekler, OpenCV BGRA formatındadır, dönüşüm yap
    plt.imshow(cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA))
    plt.title("Arka Planı Silinmiş Görüntü (ROI)")
    plt.axis('off')
    plt.show()

# --- ANA PROGRAM ---

# 1. Görüntüyü Yükle
# LÜTFEN 'resminizin_yolu.jpg' kısmını kendi resim dosyanızın yolu ile değiştirin
image_path = '/home/huawei/Documents/pepsi_ürünler-20251021T122823Z-1-001/pepsi_ürünler/000 Ben/Pepsi/Pepsi Zero 1.5 lt.jpg'
try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Dosya bulunamadı veya okunamadı: {image_path}")
    img_copy = img.copy() # Üzerinde çizim yapmak için kopya al
    print(f"'{image_path}' yüklendi.")
except Exception as e:
    print(e)
    print("Lütfen 'image_path' değişkenini geçerli bir resim yolu ile güncelleyin.")
    exit()

# 2. OpenCV penceresi oluştur ve fare olaylarını bağla
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

print("\n--- Kullanım Talimatları ---")
print("1. Fare ile ilgilendiğiniz nesnenin etrafına bir DİKDÖRTGEN çizin.")
print("2. Çizimi bitirdikten sonra 'n' (next) tuşuna basarak arka planı silin.")
print("3. Çizimi sıfırlamak için 'r' (reset) tuşuna basın.")
print("4. Çıkmak için 'q' (quit) tuşuna basın.")

while(1):
    cv2.imshow('image', img_copy)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'): # 'r' tuşu (reset)
        print("Dikdörtgen sıfırlandı.")
        img_copy = img.copy()
        rect = (0, 0, 0, 0)
        
    elif key == ord('n'): # 'n' tuşu (next)
        if rect[2] > 0 and rect[3] > 0: # Geçerli bir dikdörtgen çizildiyse
            break # Döngüden çık ve GrabCut'ı çalıştır
        else:
            print("Lütfen önce bir dikdörtgen çizin.")
            
    elif key == ord('q'): # 'q' tuşu (quit)
        print("Programdan çıkılıyor.")
        cv2.destroyAllWindows()
        exit()

# 3. Pencereyi kapat ve GrabCut'ı çalıştır
cv2.destroyAllWindows()
run_grabcut()
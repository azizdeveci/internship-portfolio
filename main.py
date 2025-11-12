import cv2
from PIL import Image
import pytesseract
import re  


def clean_ocr_text(text):
 
    pattern = r"[^a-zA-Z0-9çÇğĞıIİöÖşŞüÜ\s.,!?;:'\"\-()\[\]{}]"
    
    # Eşleşen (istenmeyen) karakterleri bir boşlukla değiştiriyoruz
    # (Kelimelerin birbirine yapışmasını önlemek için "" yerine ' ' kullanmak daha güvenlidir)
    filtered_text = re.sub(pattern, ' ', text)
    
    # Birden fazla (yatay) boşluğu tek boşluğa indirge
    filtered_text = re.sub(r' +', ' ', filtered_text)
    
    # 3 veya daha fazla yeni satırı 2 yeni satıra indirge (paragraf boşluğu olarak)
    filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
    
    # Metni satırlara böl, her birinin başını/sonunu temizle, sonra tekrar birleştir.
    filtered_text = "\n".join([line.strip() for line in filtered_text.split('\n')])
    
    # Son olarak, tüm metnin başındaki ve sonundaki fazlalıkları al
    return filtered_text.strip()


pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

IMAGE_PATH = '/home/huawei/Documents/internship/Tasks/ocr/1.jpg'

try:
    
    img = Image.open(IMAGE_PATH)

    
    print("--- Basit Metin Çıktısı (Ham) ---")
    # Önce ham metni alıyoruz
    raw_text = pytesseract.image_to_string(img, lang='tur+eng', timeout=2)
    print(raw_text)

    # --- Filtrelenmiş Çıktı (Yeni Eklenen Kısım) ---
    print("\n--- Filtrelenmiş Metin Çıktısı ---")
    # Temizleme fonksiyonumuzu ham metne uyguluyoruz
    filtered_text = clean_ocr_text(raw_text)
    print(filtered_text)


    # List of available languages
    print("\n--- Mevcut Diller ---")
    # 'config' parametresi 'get_languages' için geçerli değildir. Doğrudan çağrılır.
    print(pytesseract.get_languages()) # Sadece yüklü tüm dilleri listeler

    print("\n--- PDF ve BOX Çıktıları ---")

    # Get a searchable PDF
    try:
        pdf = pytesseract.image_to_pdf_or_hocr(IMAGE_PATH, extension='pdf')
        with open('mektup_1_output.pdf', 'w+b') as f:
            f.write(pdf) # pdf type is bytes by default
        print("PDF dosyası (mektup_1_output.pdf) başarıyla oluşturuldu.")
    except pytesseract.pytesseract.TesseractError as e:
         print(f"PDF oluşturma hatası: {e}")


    try:

        text, boxes = pytesseract.run_and_get_multiple_output(IMAGE_PATH, extensions=['txt', 'box'])
        print("Çoklu çıktı (TXT ve BOX) başarıyla alındı.")
        
    except pytesseract.pytesseract.TesseractError as e:
         print(f"Çoklu çıktı alma hatası: {e}")


    print("\nOCR modülü başarıyla yüklendi ve işlemler tamamlandı.")

except FileNotFoundError:
    print(f"HATA: Görüntü dosyası belirtilen yolda bulunamadı: {IMAGE_PATH}")
except pytesseract.pytesseract.TesseractNotFoundError:
    print(f"HATA: Tesseract executable bulunamadı. Lütfen 'pytesseract.tesseract_cmd' yolunu kontrol edin.")
except Exception as e:
    print(f"Genel bir hata oluştu: {e}")
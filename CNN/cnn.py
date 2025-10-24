#import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, # evrişimli katman
    MaxPooling2D, # havuzlama katmanı
    Flatten, # çok boyutlu veriyi tek boyuta indirme katmanı
    Dense, # tam bağlantı katmanı
    Dropout # rastgele nöron kapatma ve overfitting'i önleme katmanı
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,  #erken durdurma
    ModelCheckpoint,  #modeli kaydetme
    ReduceLROnPlateau  #öğrenme oranını azaltma
)
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow Datasets'i import et
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__)
print("TensorFlow Datasets version:", tfds.__version__)

#veri setini yükleme
print("Veri seti yükleniyor...")
try:
    (ds_train, ds_val), ds_info = tfds.load(
        "tf_flowers",  # veri seti adı
        split=["train[:80%]", "train[80%:]"],  # eğitim ve doğrulama splitleri
        as_supervised=True,  # etiketli veri seti
        with_info=True  # veri seti bilgisi
    )
    print("Veri seti başarıyla yüklendi!")
    print("Veri seti bilgisi:", ds_info.features)
    
    # Sınıf isimlerini al
    class_names = ds_info.features['label'].names
    print("Sınıf isimleri:", class_names)
    print("Toplam sınıf sayısı:", len(class_names))
    
except Exception as e:
    print("Veri seti yükleme hatası:", e)
    exit()

# Veri seti hakkında bilgi
print("\nVeri seti istatistikleri:")
print(f"Eğitim verisi örnek sayısı: {len(list(ds_train))}")
print(f"Doğrulama verisi örnek sayısı: {len(list(ds_val))}")

# Birkaç örneği göster
print("\nİlk birkaç örnek kontrol ediliyor...")
for i, (image, label) in enumerate(ds_train.take(3)):
    print(f"Örnek {i+1}: Görüntü boyutu: {image.shape}, Etiket: {label.numpy()} ({class_names[label.numpy()]})")

# Veri ön işleme fonksiyonu
def preprocess_image(image, label):
    # Resmi yeniden boyutlandır
    image = tf.image.resize(image, [128, 128])  # Daha küçük boyut için 128x128
    # Normalizasyon
    image = image / 255.0
    return image, label

# Veri setlerini ön işleme
print("\nVeri ön işleme uygulanıyor...")
ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.batch(32).prefetch(tf.data.AUTOTUNE)

print("Veri ön işleme tamamlandı!")

# Model oluşturma
print("\nModel oluşturuluyor...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')  # Dinamik sınıf sayısı
])

# Modeli derleme
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model summary:")
model.summary()

# Callback'ler
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_flowers_model.h5', save_best_only=True, verbose=1)
]

# Modeli eğitme
print("\nModel eğitimi başlıyor...")
history = model.fit(
    ds_train,
    epochs=10,  # Başlangıç için daha az epoch
    validation_data=ds_val,
    callbacks=callbacks,
    verbose=1
)

print("Model eğitimi tamamlandı!")

# Modeli değerlendirme
print("\nModel değerlendiriliyor...")
test_loss, test_accuracy = model.evaluate(ds_val, verbose=0)
print(f"Test Doğruluğu: {test_accuracy:.4f}")
print(f"Test Kaybı: {test_loss:.4f}")

# Eğitim geçmişini görselleştirme
print("\nGrafikler oluşturuluyor...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Program başarıyla tamamlandı!")
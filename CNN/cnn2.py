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

# veri seti yükleme
(ds_train, ds_val), ds_info = tfds.load(
        "tf_flowers",  # veri seti adı
        split=["train[:80%]", "train[80%:]"],  # eğitim ve doğrulama splitleri
        as_supervised=True,  # etiketli veri seti
        with_info=True  # veri seti bilgisi
    )
print(ds_info.features) # veri seti bilgisi
print("Number of classes:", ds_info.features['label'].num_classes) # sınıf sayısı

# data augmentation + preprocessing

IMG_SIZE =(180,180)

def preprocess_train(image, label):

    image = tf.image.resize(image, IMG_SIZE)  # görüntüyü yeniden boyutlandır
    image = tf.image.random_flip_left_right(image)  # yatay çevirme
    image = tf.image.random_brightness(image, max_delta=0.1)  #parlaklık artırma
    image = tf.image.random_contrast(image, lower=0.9, upper=1.2)  # kontrast artırma
    image = tf.image.random_crop(image, size=[160,160, 3])  # rastgele kırpma
    image = tf.image.resize(image, IMG_SIZE)  # yeniden boyutlandırma
    image = tf.cast(image, tf.float32)/ 255.0  # normalize etme
    return image, label

def preprocess_val(image, label):
    image = tf.image.resize(image, IMG_SIZE)  # görüntüyü yeniden boyutlandır
    image = tf.cast(image, tf.float32)/ 255.0  # normalize etme
    return image, label

# veri setini hazırlama

ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)


# CNN modeli oluşturma

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D((2, 2)), # 2x2 max pooling

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(), # çok boyutlu veriyi tek boyuta indirme

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(ds_info.features['label'].num_classes, activation='softmax')
])

# model callbacks

callbacks = [
    # val_loss 3 epoch boyunca iyileşmezse öğrenme oranını durdurur ve en iyi ağırlıkları  yükler
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), # erken durdurma

    #val_loss 2 epoch boyunca değişmezse learning rate'i 0.2 çarpanı ile azaltır
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose= 1, min_lr= 1e-9), # öğrenme oranını azaltma

    # her epoch sonunda en iyi modeli kaydeder
    ModelCheckpoint('best_model.h5', save_best_only=True) # modeli kaydetme
]


# modeli derleme

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', # etiketler tam sayı olduğundan sparse_categorical_crossentropy kullanılır
    metrics=['accuracy']
    
    )

print(model.summary())  # model özeti

# training the model

history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=20,
    callbacks=callbacks,
    
)


# model evaluation

plt.figure(figsize=(12, 5))

# accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Model Doğruluğu')
plt.legend()    

# loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Model Kaybı')
plt.legend()

plt.tight_layout()
plt.savefig("training_validation_plots.png")
plt.show()


print("Completed...")
# uv activate
# source pose_estimation/bin/activate

import mediapipe as mp
import cv2
import numpy as np


# açı hesaplaama fonksiyonu

def calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

# mediapipe çözümleri

mp_drawing = mp.solutions.drawing_utils # frame üzerinde çizim yapmak için
mp_pose = mp.solutions.pose # poz tahmini için


# video yakalama
cap = cv2.VideoCapture('/home/huawei/Documents/internship/Tasks/yolo_project/videos/squat_test1.avi')


#kural tabanlı poz tahmini

counter = 0 # Squat sayacı
stage = None # Squat pozisyonu (up veya down)

def classify_pose(knee_angle):
    #diz açısına göre poz sınıflandırma
    if knee_angle <100:
        return "Squatting" # çömelme
    
    elif 100 <= knee_angle < 160:
        return "Lunging"# Atlama pozisyonu
    
    else: 
        return "Standing" # ayakta durma pozisyonu


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR görüntüyü RGB'ye dönüştürme
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Poz tahmini
        results = pose.process(image)

        # Görüntüyü tekrar BGR'ye dönüştürme(Opencv için)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # İlgili eklem noktalarının koordinatlarını alma
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y] #(x,y)
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Diz açısını hesaplama
            angle = calculate_angle(hip, knee, ankle)

            # Pozu sınıflandırma
            current_pose = classify_pose(angle)

            # Squat sayısını güncelleme
            if current_pose == "Squatting" and stage != "down":
                stage = "down"
            if current_pose == "Standing" and stage == "down":
                stage = "up"
                counter += 1
                print(f'Squat sayısı: {counter}')
            
            # Görüntü üzerine diz açısını ve sayacı yazdırma
            cv2.putText(image,f"Diz Acisi: {int(angle)}",(10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,f'Squat Sayisi: {counter}',(10,100),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,f'Pose: {current_pose}',(10,140),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                        

        except:
            pass

        # anahtar noktaları ve bağlantıları çizme
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )


       
        # Sonuçları gösterme
        image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow('Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("Video yükleniyor...")



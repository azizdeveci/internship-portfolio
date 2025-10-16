import cv2

#görsel alınır ve gri formata çevirilir
img = cv2.imread("flower.jpeg")
gri= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 #SIFT özellik çıkarıcı aktif edilir.
sift = cv2.SIFT_create()
anahtarnokta = sift.detect(gri,None)

#Anahtar noktalar görsel üzerinde çizdirilerek bastırılır
img=cv2.drawKeypoints(gri,anahtarnokta,img)
cv2.imshow('resim1',img)

#Eğer çizdirilen anahtar noktalar yöneleriyle beraber çizdirilmek istenirse cv2 kütüphanesi içerisindeki DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS komutu kullanılır.
img=cv2.drawKeypoints(gri,anahtarnokta,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('resim2',img)

#görsel gösterilir

cv2.waitKey(0)
cv2.destroyAllWindows()
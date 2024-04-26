# binarize_otsu.py
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
img = cv2.imread("image.jpg")
# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#普通の二値化
ret, thresh1 = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
#適応的しきい値処理
thresh2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
thresh3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
#大津の二値化
ret, thresh4 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

titles = ["GRAY", "BINARY", "MEAN_C", "GAUSSIAN_C", "OTSU"]
images = [img_gray, thresh1, thresh2, thresh3, thresh4]

plt.figure(figsize=(10, 7))
for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
plt.show()
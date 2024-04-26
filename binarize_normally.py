# binarize_normally.py
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
img = cv2.imread("image.jpg")
print(img)
# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化のしきい値を設定
threshold = 128

ret, thresh1 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TOZERO_INV)

titles = ["GRAY", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
plt.show()



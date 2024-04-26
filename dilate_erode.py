# dilate_erode.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 画像の読み込み
img = cv2.imread("image.jpg")

# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 局所的にしきい値を決定し二値化
img_binary = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
)

# 膨張/収縮処理の回数を設定
iterations = 1

# カーネル(矩形)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 膨張処理
img_dilate = cv2.dilate(img_binary, kernel, iterations=iterations)

# 収縮処理
img_erode = cv2.erode(img_binary, kernel, iterations=iterations)

# 二値化画像、膨張画像、収縮画像を表示
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
plt.title("Binary")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_dilate, cv2.COLOR_BGR2RGB))
plt.title(f"膨張処理\n回数: {iterations}")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_erode, cv2.COLOR_BGR2RGB))
plt.title(f"収縮処理\n回数: {iterations}")
plt.show()





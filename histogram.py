# histogram.py
import cv2
from matplotlib import pyplot as plt

# 画像をグレースケールで読み込み
img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# ヒストグラムをグラフ表示
plt.hist(img_gray.ravel(), 256, [0,256])
plt.show()


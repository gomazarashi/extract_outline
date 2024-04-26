# denoising_bilateral.py
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
img = cv2.imread("image.jpg")

# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 局所的にしきい値を決定し二値化
img_binary = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
)

# フィルタのサイズ
d=3

#色の標準偏差
sigmaColor=36

# 距離の標準偏差
sigmaSpace=36

# バイラテラルフィルタを適応
img_bilateral = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# 二値化画像と平均化画像を表示
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
plt.title("Binary")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2RGB))
plt.title(f"バイラテラルフィルタ\nフィルタサイズ: {d}, 色の標準偏差: {sigmaColor}, 距離の標準偏差: {sigmaSpace}")
plt.show()



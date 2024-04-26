# binarize_adaptively.py
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
img = cv2.imread("image.jpg")
# グレースケール変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化のしきい値を設定
threshold = 128

ret, thresh1 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
)
thresh3 = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2
)

titles = ["GRAY", "BINARY", "MEAN_C", "GAUSSIAN_C"]
images = [img_gray, thresh1, thresh2, thresh3]

plt.figure(figsize=(10, 7))
for i in range(4):
    #キャンパスサイズを設定
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
plt.show()

cv2.imwrite("mean_c.png",thresh2)
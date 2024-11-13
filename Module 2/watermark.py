import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (15, 15)

img = cv2.imread('./orange.jpg', cv2.IMREAD_UNCHANGED)
marking = cv2.imread('./img.png', cv2.IMREAD_UNCHANGED)

plt.figure(figsize=(15,10))
plt.subplot(121); plt.imshow(img[:, :, ::-1]); plt.title('Image')
plt.subplot(122); plt.imshow(marking);       plt.title('Watermark')
plt.show()
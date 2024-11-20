import cv2 
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

img = cv2.imread('forest.jpg', cv2.IMREAD_COLOR)

# --- Histogram analysis for accurage RGB masking for green

# histB = cv2.calcHist([img], [0], None, [256], [0,255])
# histG = cv2.calcHist([img], [1], None, [256], [0,255])
# histR = cv2.calcHist([img], [2], None, [256], [0,255])

# fig, axs = plt.subplots(3)
# fig.suptitle("Forest histogram images")
# fig.tight_layout()

# axs[0].plot(histB)
# axs[0].set_title('Blue')

# axs[1].plot(histG)
# axs[1].set_title('Green')

# axs[2].plot(histR)
# axs[2].set_title('Red')
# plt.subplots_adjust(wspace=0.5, hspace=1)

# --- From histogram analysis, a majority of the Green values are between 40 to 60 

BGR_lb = np.array([0, 40, 0], np.uint8)
BGR_ub = np.array([255, 70, 255], np.uint8)

mask_BGR = cv2.inRange(img, BGR_lb, BGR_ub)

# Calculating percentage of forest using mask

c = cv2.countNonZero(mask_BGR)

t = img.shape[0] * img.shape[1]

percentage = round((c / t) * 100, 2)

fig = plt.figure(figsize = (20, 20))

ax = fig.add_subplot(1, 2, 1)
plt.imshow(img[:, :, ::-1])
ax.set_title('Original')

ax = fig.add_subplot(1, 2, 2)
plt.imshow(mask_BGR, cmap = 'gray')
ax.set_title('Color Segmented in Green Channel: ' + str(percentage) + '%')

plt.show()

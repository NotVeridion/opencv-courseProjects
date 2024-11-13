import cv2
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (15, 15)

img = cv2.imread('./orange.jpg', cv2.IMREAD_UNCHANGED)
mark = cv2.imread('./img.png', cv2.IMREAD_UNCHANGED)

orange_original = img.copy()

print(img.shape)
print(mark.shape)

mark = cv2.resize(mark, None, fx = 0.1, fy = 0.1, interpolation=cv2.INTER_LINEAR)

# Grab dimensions of images
img_h, img_w, _= img.shape
mark_h, mark_w, _= mark.shape

# plt.figure(figsize=(15,10))
# plt.subplot(121); plt.imshow(img[:, :, ::-1]); plt.title('Image')
# plt.subplot(122); plt.imshow(mark);       plt.title('Watermark')
# plt.show()

# Finding center of the image to place watermark
cx = int(img_w / 2)
cy = int(img_h / 2)

# Top left of region of interest
tpl_x = int(cx - mark_w / 2)
tpl_y = int(cy - mark_h / 2)

# Bottom right of region of interest
btr_x = int(cx + mark_w / 2)
btr_y = int(cy + mark_h / 2)

# ROI from original image
roi = img[tpl_y:btr_y, tpl_x:btr_x]

# Need a 3-channel mask of just the alpha values of the watermark png
mark_alpha = mark[:,:, 3]
mark_mask = cv2.merge([mark_alpha, mark_alpha, mark_alpha])

# Getting black cutout of mark onto roi
mark_mask_inv = cv2.bitwise_not(mark_mask)
masked_roi = cv2.bitwise_and(roi, mark_mask_inv)

# Mark with black background and color preserved
# Using [:,:,:0:3] to match sizes of channels
masked_mark = cv2.bitwise_and(mark[:,:, 0:3], mark_mask)

# Create final roi with logo imprinted
final_roi = cv2.bitwise_or(masked_roi, masked_mark)

# Now that we have the final roi with the watermark applied
# We can replace the region of the original image with it
img[tpl_y:btr_y, tpl_x:btr_x] = final_roi

orange_marked = img.copy()
cv2.imwrite('orange_watermarked.jpg', orange_marked)

plt.figure(figsize=(15,10))
plt.subplot(121); plt.imshow(orange_original[:, :, ::-1]); plt.title('Original')
plt.subplot(122); plt.imshow(orange_marked[:, :, ::-1]);       plt.title('Watermarked')
plt.show()
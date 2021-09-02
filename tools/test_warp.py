import cv2
import numpy as np


def warp(img, direction):
    h, w = img.shape[:2]
    # Direction: + for right, - for left
    original_corner_pts = np.array(
        (
            (0, 0),  # Top-left
            (w, 0),  # Top-right
            (0, h),  # Bottom-left
            (w, h),  # Bottom-right
        ), dtype=np.float32
    )

    alpha = 1
    beta = 5

    transform_offset = np.array(
        (
            (alpha * direction, 0),
            (alpha * direction, 0),
            (- beta * direction, 0),
            (- beta * direction, 0),
        ), dtype=np.float32
    )

    transformed_corner_pts = original_corner_pts + transform_offset

    transform_matrix = cv2.getPerspectiveTransform(original_corner_pts, transformed_corner_pts)

    transformed_image = cv2.warpPerspective(img, transform_matrix, (1600, 900))

    return transformed_image


# img = cv2.imread('/data/nuscenes_all/samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg')
img = cv2.imread('/data/nuscenes_mini/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800382862460.jpg')
cv2.imshow('original_img', img)


for img_idx, distort_rate in enumerate(list(range(0, -100, -1)) + list(range(-100, 100)) + list(range(100, 0, -1))):
    warp_img = warp(img, distort_rate)
    cv2.imwrite('vis_warp/%04d.jpg' % img_idx, warp_img)

    # Crop Center
    h, w, _ = warp_img.shape
    cy, cx = h // 2, w // 2
    crop_img = warp_img[int(cy-0.3*h):int(cy+0.3*h), int(cx-0.3*w):int(cx+0.3*w), :]
    cv2.imwrite('vis_warp_crop/%04d.jpg' % img_idx, crop_img)


# Keyboard demo
distort_rate = 0
while True:
    warp_img = warp(img, distort_rate)
    print(distort_rate)
    cv2.imshow('warp_img', warp_img)

    # Crop Center
    h, w, _ = warp_img.shape
    cy, cx = h // 2, w // 2
    crop_img = warp_img[int(cy-0.3*h):int(cy+0.3*h), int(cx-0.3*w):int(cx+0.3*w), :]
    cv2.imshow('crop_img', crop_img)

    k = cv2.waitKey(0)
    if k == ord('a'):                       # toggle current image
        distort_rate -= 1
    elif k == ord('d'):
        distort_rate += 1
    elif k == 27 or k == ord('q'):  #escape key 
        break

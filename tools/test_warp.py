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

    alpha = 10
    beta = 50

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


img = cv2.imread('/data/nuscenes_all/samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg')
cv2.imshow('original_img', img)

distort_rate = 0
while True:
    warp_img = warp(img, distort_rate)
    print(distort_rate)
    cv2.imshow('warp_img', warp_img)

    k = cv2.waitKey(0)
    if k == ord('a'):                       # toggle current image
        distort_rate -= 1
    elif k == ord('d'):
        distort_rate += 1
    elif k == 27 or k == ord('q'):  #escape key 
        break

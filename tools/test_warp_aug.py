import cv2
import numpy as np


def random_warp(img):
    h, w = img.shape[:2]

    random_rate = 0.1
    width_max = random_rate * w
    height_max = random_rate * h

    # 8 offsets
    w_offsets = list(np.random.uniform(0, width_max) for _ in range(4))
    h_offsets = list(np.random.uniform(0, height_max) for _ in range(4))

    print(w_offsets, h_offsets)

    original_corner_pts = np.array(
        (
            (w_offsets[0], h_offsets[0]),
            (w - w_offsets[1], h_offsets[1]),
            (w_offsets[2], h - h_offsets[2]),
            (w - w_offsets[3], h - h_offsets[3]),
        ), dtype=np.float32
    )

    target_corner_pts = np.array(
        (
            (0, 0),  # Top-left
            (w, 0),  # Top-right
            (0, h),  # Bottom-left
            (w, h),  # Bottom-right
        ), dtype=np.float32
    )

    transform_matrix = cv2.getPerspectiveTransform(original_corner_pts, target_corner_pts)

    transformed_image = cv2.warpPerspective(img, transform_matrix, (w, h))

    return transformed_image


while True:
    # img = cv2.imread('/data/nuscenes_all/samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg')
    img = cv2.imread('/data/nuscenes_mini/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800382862460.jpg')
    cv2.imshow('original_img', img)


    warp_img = random_warp(img)
    cv2.imshow('warp_img', warp_img)

    # Crop Center
    h, w, _ = warp_img.shape
    cy, cx = h // 2, w // 2
    crop_img = warp_img[int(cy-0.3*h):int(cy+0.3*h), int(cx-0.3*w):int(cx+0.3*w), :]
    cv2.imshow('crop_img', crop_img)

    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):  #escape key 
        break

import numpy as np
import cv2

frame = np.zeros((874, 1164, 3))

def angle_diff(a, b):
    return np.pi - abs(abs(a - b) - np.pi)

def get_angle_to_center(frame):
    h, w = frame.shape[:2]
    w1 = np.expand_dims(np.arange(float(w)), axis=0)
    w2 = np.repeat(w1, h, 0)
    h1 = np.expand_dims(np.arange(float(h)), axis=0)
    h2 = np.repeat(h1, w ,0).T

    pixel_map = np.dstack((w2,h2))
    center_x = w / 2.0
    center_y = h / 2.0

    pixel_map[..., 0] = pixel_map[..., 0] - center_x
    pixel_map[..., 1] = center_y - pixel_map[..., 1]
    return np.arctan2(pixel_map[..., 1] , pixel_map[..., 0]) + np.pi

frame[..., 0] = get_angle_to_center(frame) / np.pi / 2 * 180
frame[..., 1] = 255
frame[..., 2] = 255

frame = frame.astype(np.uint8)
rgb = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
cv2.imwrite("test.png", rgb)

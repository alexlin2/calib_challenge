import cv2 
import os
import numpy as np
import glob
from util import get_angle_to_center, angle_diff

data_dir = "/home/alexlin/Developer/calib_challenge/labeled"
frame_dir = "/home/alexlin/Developer/calib_challenge/images_labeled"

vid_dir = sorted(glob.glob(data_dir + "/*.hevc"))

cap = cv2.VideoCapture(vid_dir[3])
success, frame = cap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(frame.shape)

mask = np.zeros_like(frame)
mask[..., 1] = 255

angle_to_center = get_angle_to_center(frame)

while cap.isOpened():
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    corrected_angle = np.pi - angle

    mask[..., 0] = corrected_angle / np.pi / 2 * 180
    mask[..., 2] = np.where(angle_diff(corrected_angle, angle_to_center) < 0.1, \
                        cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX), 0)

    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    prev_gray = gray

    cv2.imshow("frame", rgb)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destoryAllWindows()
    



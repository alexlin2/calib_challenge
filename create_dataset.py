import cv2
import os
import glob

data_dir = "/home/alexlin/calib_challenge/labeled"
img_dir = "/home/alexlin/calib_challenge/images_labeled"

for idx, name in enumerate(sorted(glob.glob(data_dir + "/*.hevc"))):
    cap = cv2.VideoCapture(name)
    success, img = cap.read()
    frame_idx = 0
    while success:
        filename = img_dir + "/frame{}_{:05d}.png".format(idx, frame_idx)
        print(filename)
        cv2.imwrite(filename, img)
        success, img = cap.read()
        frame_idx += 1
    cap.release()


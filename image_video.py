import cv2
import os

img_root = "./show"
fps = 30
size = (515, 330)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('show.avi', fourcc, fps, size)
for x in sorted(os.listdir(img_root), key=lambda x: [int(x.split("_")[0]), x.split("_")[1]]):
    frame = cv2.imread(os.path.join(img_root, x))
    videoWriter.write(frame)
videoWriter.release()

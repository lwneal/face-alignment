import sys
import time
import imutil
import face_alignment
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

if len(sys.argv) < 2:
    print('Usage: main.py <input.mp4>')
    exit()

def read_images():
    cap = cv2.VideoCapture(sys.argv[1])
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame[:,:,::-1]





fig = plt.figure(figsize=(6.4, 6.4))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(10, 0)
fig.tight_layout(rect=[0, 0.01, 1, 0.99])

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D)

vid = imutil.Video('output_{}.mp4'.format(int(time.time())))

for img in read_images():
    start_time = time.time()
    preds = fa.get_landmarks(img)[0]
    print('Timing: {:.02f} seconds for one frame'.format(time.time() - start_time))
    import pdb; pdb.set_trace()
    #ax.set_xlim3d(-500, 500)
    #ax.set_ylim3d(-500, 500)
    #ax.set_zlim3d(-200, 200)
    ax.scatter(preds[:, 2], preds[:, 0], -preds[:, 1])
    left = imutil.get_pixels(img, 640, 640)
    right = imutil.get_pixels(plt, 640, 640)
    pixels = np.concatenate([left, right], axis=1)
    vid.write_frame(pixels)
    imutil.show(pixels, save=False)
    ax.clear()

vid.finish()

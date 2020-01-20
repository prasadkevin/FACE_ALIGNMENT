import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2, glob
import datetime
import imutils
from matplotlib import pyplot as plt
import numpy as np
 
# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

video_path = 'youtube_video/video.mp4'
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)

print("Running on video >> {}".format(video_path))

start = datetime.datetime.now()
frame_count = 0
for i in range(1123):
    ret, frame = cap.read()
    frame_count += 1
    
while True:
   
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        print('not grabbed')
        break
    if frame_count % 1 == 0:
        print(frame_count)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 0xFF:
                    break
                
        input_img = imutils.resize(frame, width=450)
    #    io.imread('test/assets/aflw-test.jpg')
            
        xx = fa.get_landmarks(input_img)
        if xx == None:
            continue
        
        preds = xx[-1]
        
        # 2D-Plot
        plot_style = dict(marker='o',
                          markersize=4,
                          linestyle='-',
                          lw=2)
        
        pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
        pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                      'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                      'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                      'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                      'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                      'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                      'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                      'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                      'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                      }
        
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input_img)
        
#        cv2.imshow('xx', input_img)
        key = cv2.waitKey(1) & 0xFF
    
    
        for pred_type in pred_types.values():
            ax.plot(preds[pred_type.slice, 0],
                    preds[pred_type.slice, 1],
                    color=pred_type.color, **plot_style)
        
        ax.axis('off')
        
        # 3D-Plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(preds[:, 0] * 1.2,
                          preds[:, 1],
                          preds[:, 2],
                          c='cyan',
                          alpha=1.0,
                          edgecolor='b')
        
        for pred_type in pred_types.values():
            ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                      preds[pred_type.slice, 1],
                      preds[pred_type.slice, 2], color='blue')
        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
#        plt.show()
        fig.set_size_inches(50.5, 30.5)
        fig.savefig('images/test_{}.png'.format(frame_count))
        plt.clf()
    
cap.release()
#cv2.destroyAllWindows()





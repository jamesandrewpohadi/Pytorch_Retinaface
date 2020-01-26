from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from models.retinaface import SinvNet
from utils.retinaface import load_model, detect
import cv2
import numpy as np
import torch

cfg = cfg_re50
cfg['min_sizes']= [[12, 16],[24, 32],[48, 64],[96, 128],[192, 256],[256,512]]
cfg['steps'] = [8, 16, 32,64,128,256]
model = SinvNet(cfg=cfg,fpn_level=6)
load_model(model,'weights/SinvNet_Resnet50_epoch_17.pth')
model.eval()
model = model.to(torch.device('cuda'))
print('finished loading model!')
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    
    dets = detect(model,frame,cfg,(200,400))
    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(frame, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)

    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
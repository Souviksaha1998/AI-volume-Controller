import cv2 
import mediapipe as mp
import time
from handtrackingmodule import HandDetector 
import math
import numpy as np
from subprocess import call #https://askubuntu.com/questions/689521/control-volume-using-python-script


cap = cv2.VideoCapture(0)
hd = HandDetector()
ptime = 0
while True:
    ret , frame = cap.read()
    hands = hd.findHand(frame,draw=False)
    handpos = hd.findHandPosition(frame)
    if len(handpos) != 0:
        # print(handpos[4])
        cx1 , cy1 = handpos[4][1] , handpos[4][2]
        cx2 , cy2 = handpos[8][1] , handpos[8][2]
        cx3 , cy3 = handpos[12][1] , handpos[12][2]
        cx , cy = (cx1+cx2)//2 , (cy1+cy2)//2
        length = math.hypot(cx2-cx1,cy2-cy1)
        distance = math.hypot(cx3-cx1,cy3-cy1)
        # print(int(distance))
        convert = np.interp(length,[10,160],[0,100])
        # print(int(convert))
        cv2.circle(frame,(cx1,cy1),5,(255,0,255),-1)
        cv2.circle(frame,(cx2,cy2),5,(255,0,255),-1)
        cv2.line(frame,(cx1,cy1),(cx2,cy2),(255,255,255),3)
        cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
        # cv2.line(frame,(cx2,cy2),(cx3,cy3),(0,0,255),2)
        cv2.circle(frame,(cx3,cy3),5,(255,255,0),-1)
        max_distance = 10
        if distance<max_distance:
            call(["amixer", "-D", "pulse", "sset", "Master", str(int(convert))+"%"])
            cv2.putText(frame,f'VOLUME :{int(convert)}%',(cx+10,cy),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

            if length>100:
                cv2.circle(frame,(cx,cy),10,(255,255,255),-1)
                cv2.putText(frame,'MAX DISTANCE',(cx1,cy1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            
            elif length<15:
                cv2.circle(frame,(cx,cy),10,(0,255,255),-1)
           

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f'FPS : {str(int(fps))}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('video' , hands)
    if cv2.waitKey(1)  == 27:
        break

cap.release()
cv2.destroyAllWindows()
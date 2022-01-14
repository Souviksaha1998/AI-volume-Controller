import cv2
import mediapipe as mp
import time


hands_detect = mp.solutions.hands
hands = hands_detect.Hands(static_image_mode=True,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5,)

draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(11,30)
ptime = 0

while True:
    ret , frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_process = hands.process(imgRGB)
    lmlist = []
    if img_process.multi_hand_landmarks:
        
        for i in img_process.multi_hand_landmarks: 
            print(i.landmark)
            for id,lm in enumerate(i.landmark):
                # print(id,lm)
                h,w,c = frame.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                if id==0 or id==10:
                    lmlist.append([id,cx,cy])
                    print(lmlist)
                    cv2.circle(frame,(cx,cy),10,(255,0,255),cv2.FILLED)
            # draw.draw_landmarks(frame,i,hands_detect.HAND_CONNECTIONS,
            # draw.DrawingSpec((0,0,255),3,2),
            # draw.DrawingSpec((0,255,0),3,2))
            

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f'FPS : {str(int(fps))}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('imge',frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
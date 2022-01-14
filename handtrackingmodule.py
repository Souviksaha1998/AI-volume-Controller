import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self,static_image_mode=True,max_num_hands=2,modelComplexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5,):
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detCon = min_detection_confidence
        self.detHands = min_tracking_confidence
        self.hands_detect = mp.solutions.hands
        self.complex = modelComplexity
        self.hand = self.hands_detect.Hands(self.mode,self.maxHands,self.complex,self.detCon,self.detHands)
        self.draw = mp.solutions.drawing_utils

    def findHand(self,frame,draw=True): 
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_process = self.hand.process(imgRGB)
        if self.img_process.multi_hand_landmarks:
            for i in self.img_process.multi_hand_landmarks:
                if draw==True:
                    self.draw.draw_landmarks(frame,i,self.hands_detect.HAND_CONNECTIONS,self.draw.DrawingSpec((0,0,255),3,2),self.draw.DrawingSpec((0,255,0),3,2))
            
        return frame

    def findHandPosition(self,img,handNo=0,draw=False):
        landmarks = []
        if self.img_process.multi_hand_landmarks:
            hand = self.img_process.multi_hand_landmarks[handNo]
            for id,lm in enumerate(hand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                landmarks.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),20,(255,0,255),cv2.FILLED)
        return landmarks

def main_():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    detector = HandDetector()
    ptime = 0
    while True:
        _ , frame = cap.read()
        frame = detector.findHand(frame)
        lmlist = detector.findHandPosition(frame)
        if len(lmlist) != 0:
            print(lmlist[4])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(frame,f'FPS : {str(int(fps))}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('imge',frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ =='__main__':
    main_()
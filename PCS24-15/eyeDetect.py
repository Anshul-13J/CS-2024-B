import cv2 as cv
import numpy as np
import mediapipe as mp
# from cvzone.SerialModule import SerialObject
import pyrebase
from datetime import datetime
import time

isDrowsy = False
config = {
    "apiKey": "AIzaSyA8qt2qq_Knpp4nOUVcBgSYueOwRKCSlfc",
    "authDomain": "test-6cdd9.firebaseapp.com",
    "databaseURL": "https://test-6cdd9-default-rtdb.firebaseio.com",
    "projectId": "test-6cdd9",
    "storageBucket": "test-6cdd9.appspot.com",
    "messagingSenderId": "830231640135",
    "appId": "1:830231640135:web:b64f2c232c5cdcb5fd5962",
}
firebase = pyrebase.initialize_app(config)
database = firebase.database()

def open_len(arr):
    y_arr = []

    for _,y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
# arduino = SerialObject()
start_time = time.time()
cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    drowsy_frames = 0

    max_left = 0
    max_right = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            right_eye = all_landmarks[RIGHT_EYE]
            left_eye = all_landmarks[LEFT_EYE]

            cv.polylines(frame, [left_eye], True, (0,255,0), 1, cv.LINE_AA)
            cv.polylines(frame, [right_eye], True, (0,255,0), 1, cv.LINE_AA) 

            len_left = open_len(right_eye)
            len_right = open_len(left_eye)

            if len_left > max_left:
                max_left = len_left

            if len_right > max_right:
                max_right = len_right

            cv.putText(img=frame, text='Max: ' + str(max_left)  + ' Left Eye: ' + str(len_left), fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
            cv.putText(img=frame, text='Max: ' + str(max_right)  + ' Right Eye: ' + str(len_right), fontFace=0, org=(10, 50), fontScale=0.5, color=(0, 255, 0))

            if (len_left <= int(max_left / 2) + 1 and len_right <= int(max_right / 2) + 1):
                drowsy_frames += 1
            else:
                drowsy_frames = 0
            if (drowsy_frames > 20):
                isDrowsy = True
                # arduino.sendData([2])
                
                cv.putText(img=frame, text='ALERT', fontFace=0, org=(200, 300), fontScale=3, color=(0, 0, 255), thickness = 4)
            else:
                pass
                # arduino.sendData([1])
                isDrowsy = False

           
            if time.time() - start_time >= 10:
            # if True:
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                data = {
                        "logStat" : date_time,
                        "isDrowsy": isDrowsy,
                        "message" : "Take a Rest" if isDrowsy else "Going Good",
                }
                database.child("Drivers").child("Driver1").push(data)
                print("Data logged successfully")
                print(time.time() - start_time)
                start_time = time.time()


        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()